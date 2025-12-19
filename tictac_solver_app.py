import csv
import json
import os
import random
import uuid
from datetime import datetime
from math import comb
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="LLM vs LLM Tic Tac Toe", layout="wide")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_LLM_A_MODEL = os.getenv("TICTAC_LLM_A_MODEL", "gpt-3.5-turbo")
DEFAULT_LLM_B_MODEL = os.getenv("TICTAC_LLM_B_MODEL", "gpt-4o-mini")
FALLBACK_MODEL_LIBRARY = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-3.5-turbo",
    "o4-mini",
]
LLM_A_MODEL = DEFAULT_LLM_A_MODEL
LLM_B_MODEL = DEFAULT_LLM_B_MODEL
def _friendly_label(model_name: str) -> str:
    label = model_name or "Model"
    if label.startswith("gpt-"):
        label = label[4:]
    return label.replace("-", " ").strip()

LLM_A_LABEL = os.getenv("TICTAC_LLM_A_LABEL", _friendly_label(LLM_A_MODEL))
LLM_B_LABEL = os.getenv("TICTAC_LLM_B_LABEL", _friendly_label(LLM_B_MODEL))
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

AGENTS = {
    "A": {"name": LLM_A_LABEL, "symbol": "X", "model": LLM_A_MODEL},
    "B": {"name": LLM_B_LABEL, "symbol": "O", "model": LLM_B_MODEL},
}


def minimax_fee_details(agent_key: str) -> Dict[str, int]:
    """Compute the current minimax entry fee and its components."""
    base_fee = 10
    opponent_key = "B" if agent_key == "A" else "A"
    points_agent = st.session_state.tournament_points.get(agent_key, 0)
    points_opponent = st.session_state.tournament_points.get(opponent_key, 0)
    score_gap = points_agent - points_opponent
    deficit = max(0, -score_gap)
    discount_steps = min(9, deficit // 10)
    adjustment = -discount_steps
    dynamic_fee = max(1, min(10, base_fee + adjustment))
    return {
        "fee": dynamic_fee,
        "base_fee": base_fee,
        "adjustment": adjustment,
        "score_gap": score_gap,
        "agent_points": points_agent,
        "opponent_points": points_opponent,
    }
DEFAULT_MAX_GAMES = 20
GAME_LOG_FILE = "game_history.csv"
CHEAP_MODEL_ALLOWLIST = {
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4.1-mini",
    "o4-mini",
    "gpt-3.5-turbo",
}
MAX_LLM_RETRIES = 3
MAX_GAME_RETRIES = 3
OPENING_SEQUENCE = [0, 2, 6, 8, 4, 1, 3, 5, 7]

WINNING_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]

WIN_LINE_CLASS_MAP = {
    (0, 1, 2): "line-h0",
    (3, 4, 5): "line-h1",
    (6, 7, 8): "line-h2",
    (0, 3, 6): "line-v0",
    (1, 4, 7): "line-v1",
    (2, 5, 8): "line-v2",
    (0, 4, 8): "line-d0",
    (2, 4, 6): "line-d1",
}


def get_winning_line(board: List[str]) -> Optional[Tuple[int, int, int]]:
    for line in WINNING_LINES:
        a, b, c = line
        if board[a] and board[a] == board[b] == board[c]:
            return line
    return None


def check_winner(board: List[str]) -> Optional[str]:
    winning_line = get_winning_line(board)
    if winning_line:
        return board[winning_line[0]]
    if "" not in board:
        return "Draw"
    return None


def minimax(
    board: List[str],
    current: str,
    ai_symbol: str,
    human_symbol: str,
) -> Tuple[int, Optional[int]]:
    """Retained minimax search for perfect play."""
    winner = check_winner(board)
    if winner:
        if winner == ai_symbol:
            return 1, None
        if winner == human_symbol:
            return -1, None
        return 0, None

    best_score = float("-inf") if current == ai_symbol else float("inf")
    best_move: Optional[int] = None

    for i in range(9):
        if board[i]:
            continue
        board[i] = current
        score, _ = minimax(
            board,
            "O" if current == "X" else "X",
            ai_symbol,
            human_symbol,
        )
        board[i] = ""

        if current == ai_symbol and score > best_score:
            best_score = score
            best_move = i
        if current != ai_symbol and score < best_score:
            best_score = score
            best_move = i

    return best_score, best_move


def binomial_two_tailed_p(successes: int, trials: int, p: float = 0.5) -> float:
    """Compute a two-tailed binomial p-value against a fairness null."""
    if trials <= 0:
        return 1.0
    successes = max(0, min(trials, successes))
    prob_success = comb(trials, successes) * (p**successes) * ((1 - p) ** (trials - successes))
    threshold = prob_success + 1e-12
    total = 0.0
    for k in range(trials + 1):
        prob = comb(trials, k) * (p**k) * ((1 - p) ** (trials - k))
        if prob <= threshold:
            total += prob
    return min(1.0, total)


def ensure_state() -> None:
    if "board" not in st.session_state:
        reset_board()
    if "current_player" not in st.session_state:
        st.session_state.current_player = "A"
    if "running" not in st.session_state:
        st.session_state.running = False
    if "game_number" not in st.session_state:
        st.session_state.game_number = 1
    if "scoreboard" not in st.session_state:
        st.session_state.scoreboard: List[Dict] = []
    if "minimax_flags" not in st.session_state:
        st.session_state.minimax_flags = {"A": False, "B": False}
    if "llm_status" not in st.session_state:
        st.session_state.llm_status = ""
    if "starting_player" not in st.session_state:
        st.session_state.starting_player = "A"
    if "opening_index" not in st.session_state:
        st.session_state.opening_index = 0
    if "opening_move_done" not in st.session_state:
        st.session_state.opening_move_done = False
    if "opening_seed" not in st.session_state:
        st.session_state.opening_seed = None
    if "opening_sequence" not in st.session_state:
        refresh_opening_sequence()
    if "current_tokens" not in st.session_state:
        st.session_state.current_tokens = {"A": 0, "B": 0}
    if "moves_this_game" not in st.session_state:
        st.session_state.moves_this_game: List[Dict] = []
    if "move_history" not in st.session_state:
        st.session_state.move_history: List[Dict] = []
    if "tournament_points" not in st.session_state:
        st.session_state.tournament_points = {"A": 0, "B": 0}
    if "model_a_choice" not in st.session_state:
        st.session_state.model_a_choice = DEFAULT_LLM_A_MODEL
    if "model_b_choice" not in st.session_state:
        st.session_state.model_b_choice = DEFAULT_LLM_B_MODEL
    if "allow_minimax" not in st.session_state:
        st.session_state.allow_minimax = True
    if "available_models" not in st.session_state:
        st.session_state.available_models = []
    if "pregame_decided" not in st.session_state:
        st.session_state.pregame_decided = False
    if "minimax_fee_paid" not in st.session_state:
        st.session_state.minimax_fee_paid = {"A": 0, "B": 0}
    if "minimax_usage_count" not in st.session_state:
        st.session_state.minimax_usage_count = {"A": 0, "B": 0}
    if "game_retry_count" not in st.session_state:
        st.session_state.game_retry_count = 0
    if "max_games" not in st.session_state:
        st.session_state.max_games = DEFAULT_MAX_GAMES
    if "game_incidents" not in st.session_state:
        st.session_state.game_incidents: Dict[int, List[str]] = {}
    if "series_id" not in st.session_state:
        st.session_state.series_id = str(uuid.uuid4())
    if "series_logged" not in st.session_state:
        st.session_state.series_logged = False
    if "human_agents" not in st.session_state:
        st.session_state.human_agents = {"A": False, "B": False}
    if "human_pending_move" not in st.session_state:
        st.session_state.human_pending_move = {"A": None, "B": None}
    if "winning_line" not in st.session_state:
        st.session_state.winning_line = None
    if "last_finished_board" not in st.session_state:
        st.session_state.last_finished_board: Optional[List[str]] = None
    if "last_finished_line" not in st.session_state:
        st.session_state.last_finished_line: Optional[Tuple[int, int, int]] = None
    if "last_finished_game" not in st.session_state:
        st.session_state.last_finished_game: Optional[int] = None


def reset_board() -> None:
    st.session_state.board = [""] * 9
    st.session_state.current_player = st.session_state.get("starting_player", "A")
    st.session_state.minimax_flags = {"A": False, "B": False}
    st.session_state.llm_status = ""
    st.session_state.opening_move_done = False
    st.session_state.current_tokens = {"A": 0, "B": 0}
    st.session_state.moves_this_game = []
    st.session_state.pregame_decided = False
    st.session_state.minimax_fee_paid = {"A": 0, "B": 0}
    st.session_state.winning_line = None
    for agent_key in ("A", "B"):
        st.session_state.human_pending_move[agent_key] = None


def reset_series() -> None:
    st.session_state.scoreboard = []
    st.session_state.game_number = 1
    st.session_state.running = False
    st.session_state.starting_player = "A"
    st.session_state.opening_index = 0
    st.session_state.move_history = []
    st.session_state.tournament_points = {"A": 0, "B": 0}
    st.session_state.minimax_usage_count = {"A": 0, "B": 0}
    st.session_state.last_finished_board = None
    st.session_state.last_finished_line = None
    st.session_state.last_finished_game = None
    refresh_opening_sequence()
    reset_board()
    st.session_state.game_retry_count = 0
    st.session_state.max_games = st.session_state.get("max_games", DEFAULT_MAX_GAMES)
    st.session_state.game_incidents = {}
    st.session_state.series_id = str(uuid.uuid4())
    st.session_state.series_logged = False


def refresh_opening_sequence(seed: Optional[int] = None) -> None:
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    rng = random.Random(seed)
    sequence = OPENING_SEQUENCE[:]
    rng.shuffle(sequence)
    st.session_state.opening_sequence = sequence
    st.session_state.opening_seed = seed


def load_model_library() -> List[str]:
    def add_unique(items: List[str], model_id: Optional[str]) -> None:
        if (
            model_id
            and model_id not in items
            and (model_id in CHEAP_MODEL_ALLOWLIST or model_id in (DEFAULT_LLM_A_MODEL, DEFAULT_LLM_B_MODEL))
        ):
            items.append(model_id)

    models = FALLBACK_MODEL_LIBRARY[:]
    add_unique(models, DEFAULT_LLM_A_MODEL)
    add_unique(models, DEFAULT_LLM_B_MODEL)
    if client:
        try:
            response = client.models.list()
            for model in getattr(response, "data", []):
                model_id = getattr(model, "id", None)
                if not model_id:
                    continue
                if model_id not in CHEAP_MODEL_ALLOWLIST:
                    continue
                add_unique(models, model_id)
        except Exception as exc:  # pragma: no cover - network call
            st.warning(f"Could not load OpenAI models dynamically: {exc}")
    st.session_state.available_models = [m for m in models if m in CHEAP_MODEL_ALLOWLIST or m in (DEFAULT_LLM_A_MODEL, DEFAULT_LLM_B_MODEL)]
    return models


def seed_opening_move() -> None:
    if st.session_state.opening_move_done:
        return
    opening_sequence = st.session_state.get("opening_sequence", OPENING_SEQUENCE)
    seq_idx = st.session_state.opening_index % len(opening_sequence)
    board_idx = opening_sequence[seq_idx]
    st.session_state.opening_index += 1

    agent_key = st.session_state.current_player
    agent = AGENTS[agent_key]
    if st.session_state.board[board_idx]:
        st.session_state.opening_move_done = True
        return

    st.session_state.board[board_idx] = agent["symbol"]
    st.session_state.llm_status = (
        f"{agent['name']} tried opening variation at cell {board_idx + 1}."
    )
    row, col = divmod(board_idx, 3)
    log_move(agent_key, row, col, "Opening")
    st.session_state.opening_move_done = True
    st.session_state.current_player = "B" if agent_key == "A" else "A"


def board_as_grid(board: List[str]) -> List[List[str]]:
    return [board[i : i + 3] for i in range(0, 9, 3)]


def render_board(
    board: List[str], winning_line: Optional[Tuple[int, int, int]] = None
) -> None:
    line_key = tuple(winning_line) if winning_line else None
    line_class = WIN_LINE_CLASS_MAP.get(line_key, "") if line_key else ""
    overlay_class = f" {line_class}" if line_class else ""
    winning_cells = set(line_key or [])
    html = """
    <style>
    .ttt-wrapper {
        display: flex;
        justify-content: center;
        margin: 1rem 0 1.5rem;
    }
    .ttt-grid-container {
        position: relative;
        display: inline-block;
    }
    .ttt-grid {
        display: grid;
        grid-template-columns: repeat(3, 120px);
        grid-template-rows: repeat(3, 120px);
        border: 6px solid #222;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    .ttt-cell {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 58px;
        font-weight: 800;
        font-family: "Inter", "Segoe UI", sans-serif;
        color: #1f1f1f;
        border-right: 4px solid #222;
        border-bottom: 4px solid #222;
        background: radial-gradient(circle at 30% 30%, #fff, #f4f4f4);
    }
    .ttt-cell:nth-child(3n) { border-right: none; }
    .ttt-cell:nth-last-child(-n+3) { border-bottom: none; }
    .ttt-cell.x { color: #ff5e5b; }
    .ttt-cell.o { color: #4b7bff; }
    .ttt-cell.win {
        background: linear-gradient(135deg, rgba(249,235,160,0.9), rgba(255,200,87,0.9));
        box-shadow: inset 0 0 10px rgba(0,0,0,0.2);
    }
    .ttt-overlay {
        position: absolute;
        top: 6px;
        left: 6px;
        width: 360px;
        height: 360px;
        pointer-events: none;
        z-index: 2;
    }
    .ttt-overlay line {
        stroke: #27ae60;
        stroke-width: 12px;
        stroke-linecap: round;
        opacity: 0;
        filter: drop-shadow(0 0 6px rgba(0,0,0,0.25));
    }
    .ttt-overlay.line-h0 .line-h0,
    .ttt-overlay.line-h1 .line-h1,
    .ttt-overlay.line-h2 .line-h2,
    .ttt-overlay.line-v0 .line-v0,
    .ttt-overlay.line-v1 .line-v1,
    .ttt-overlay.line-v2 .line-v2,
    .ttt-overlay.line-d0 .line-d0,
    .ttt-overlay.line-d1 .line-d1 {
        opacity: 1;
    }
    </style>
    <div class="ttt-wrapper">
      <div class="ttt-grid-container">
        <div class="ttt-grid">
    """
    for idx, value in enumerate(board):
        cls = ""
        if value == "X":
            cls = "x"
        elif value == "O":
            cls = "o"
        if idx in winning_cells:
            cls += " win"
        symbol = value if value else "&nbsp;"
        html += f'<div class="ttt-cell {cls}">{symbol}</div>'
    html += f"""
        </div>
        <svg class="ttt-overlay{overlay_class}" viewBox="0 0 360 360" preserveAspectRatio="none">
          <line class="line-h0" x1="10" y1="60" x2="350" y2="60" />
          <line class="line-h1" x1="10" y1="180" x2="350" y2="180" />
          <line class="line-h2" x1="10" y1="300" x2="350" y2="300" />
          <line class="line-v0" x1="60" y1="10" x2="60" y2="350" />
          <line class="line-v1" x1="180" y1="10" x2="180" y2="350" />
          <line class="line-v2" x1="300" y1="10" x2="300" y2="350" />
          <line class="line-d0" x1="20" y1="20" x2="340" y2="340" />
          <line class="line-d1" x1="340" y1="20" x2="20" y2="340" />
        </svg>
      </div>
    </div>
    """
    components.html(html, height=430, width=420)


def log_move(agent_key: str, row: int, col: int, source: str) -> None:
    st.session_state.moves_this_game.append(
        {
            "game": st.session_state.game_number,
            "agent": agent_key,
            "symbol": AGENTS[agent_key]["symbol"],
            "row": row + 1,
            "col": col + 1,
            "source": source,
        }
    )


def log_game_incident(game_number: int, message: str) -> None:
    incidents = st.session_state.game_incidents.setdefault(game_number, [])
    incidents.append(message)


def append_series_log(scoreboard: List[Dict]) -> None:
    if not scoreboard:
        return
    timestamp = datetime.utcnow().isoformat()
    base = {
        "timestamp": timestamp,
        "series_id": st.session_state.get("series_id", ""),
        "series_games_configured": st.session_state.get("max_games", DEFAULT_MAX_GAMES),
        "A_model": AGENTS["A"]["model"],
        "B_model": AGENTS["B"]["model"],
    }
    rows: List[Dict[str, Optional[str]]] = []
    for entry in scoreboard:
        row = base.copy()
        row.update(
            {
                "game": entry.get("game"),
                "winner": entry.get("winner"),
                "starter": entry.get("starter"),
                "moves": entry.get("move_count"),
                "A_used_minimax": entry.get("A_used_minimax"),
                "B_used_minimax": entry.get("B_used_minimax"),
                "A_tokens": entry.get("A_tokens"),
                "B_tokens": entry.get("B_tokens"),
                "A_points": entry.get("A_points"),
                "B_points": entry.get("B_points"),
                "incidents": " | ".join(entry.get("incidents", []) or []),
            }
        )
        rows.append(row)

    file_exists = os.path.exists(GAME_LOG_FILE)
    try:
        with open(GAME_LOG_FILE, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:
        st.warning(f"Could not append tournament log to {GAME_LOG_FILE}: {exc}")


def llm_pick_move(
    board: List[str],
    llm_symbol: str,
    opponent_symbol: str,
    agent_label: str,
    model_name: str,
) -> Tuple[Optional[Tuple[int, int]], Optional[str], int]:
    rows = board_as_grid(board)
    board_text = "\n".join(
        [" | ".join(cell if cell else "_" for cell in row) for row in rows]
    )
    available_moves = [
        (idx // 3 + 1, idx % 3 + 1) for idx, cell in enumerate(board) if not cell
    ]
    if not client:
        return None, "Set OPENAI_API_KEY to let the LLM play.", 0

    few_shots = """
Q: X _ _
   _ O _
   _ _ _
As X, best move? (Win is possible on row 1; take it.)
A: {"row": 1, "col": 3}

Q: X O X
   O X _
   _ O _
As X, best move? (Block opponent's win on column 3.)
A: {"row": 3, "col": 3}

Q: O _ _
   _ X _
   _ _ O
As X, best move? (Take center to create fork threats.)
A: {"row": 2, "col": 2}

Q: X _ _
   _ _ _
   _ _ _
As X, best move? (Open with the center when available.)
A: {"row": 2, "col": 2}

Q: O X _
   _ _ _
   _ _ X
As O, best move? (Block X's diagonal fork by playing row 2 col 2.)
A: {"row": 2, "col": 2}

Q: X O X
   O X O
   _ _ _
As X, best move? (Take a corner to threaten the final row.)
A: {"row": 3, "col": 1}
"""
    prompt = f"""
You are {agent_label} playing Tic Tac Toe as {llm_symbol}. Your opponent uses {opponent_symbol}.

Examples of perfect moves:
{few_shots.strip()}

Guidelines:
- If you can win immediately, do it.
- Otherwise, block any immediate opponent win.
- Otherwise, take the center if open.
- Otherwise, take a corner (1,1 / 1,3 / 3,1 / 3,3).
- Otherwise, play any remaining edge square.

Current board (rows use _ for empty cells):
{board_text}

Available moves (row, col): {available_moves}

Return ONLY JSON:
{{"row": <1-3>, "col": <1-3>}}
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Play legal, strong Tic Tac Toe and block losses.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
    except Exception as exc:  # pragma: no cover - network call
        return None, f"{agent_label} LLM error: {exc}", 0

    content = response.choices[0].message.content.strip()
    tokens_used = getattr(getattr(response, "usage", None), "total_tokens", 0)
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            return None, f"{agent_label} returned invalid JSON.", tokens_used
        payload = json.loads(content[start : end + 1])

    row = int(payload.get("row", -1)) - 1
    col = int(payload.get("col", -1)) - 1
    if row not in (0, 1, 2) or col not in (0, 1, 2):
        return None, f"{agent_label} chose an out-of-range move.", tokens_used
    if board[row * 3 + col]:
        return None, f"{agent_label} selected an occupied square.", tokens_used
    return (row, col), None, tokens_used


def llm_decide_minimax(agent_key: str) -> Tuple[bool, str, int]:
    agent = AGENTS[agent_key]
    opponent_key = "B" if agent_key == "A" else "A"
    opponent = AGENTS[opponent_key]
    points_agent = st.session_state.tournament_points.get(agent_key, 0)
    points_opponent = st.session_state.tournament_points.get(opponent_key, 0)
    wins_agent = sum(1 for e in st.session_state.scoreboard if e["winner"] == agent_key)
    wins_opponent = sum(
        1 for e in st.session_state.scoreboard if e["winner"] == opponent_key
    )
    draws = sum(1 for e in st.session_state.scoreboard if e["winner"] == "Draw")
    fee_info = minimax_fee_details(agent_key)
    next_fee = fee_info["fee"]
    adjustment_str = (
        f"{fee_info['adjustment']:+d}"
        if isinstance(fee_info["adjustment"], int)
        else str(fee_info["adjustment"])
    )
    score_gap = fee_info["score_gap"]
    if score_gap < 0:
        gap_phrase = f"{-score_gap} point deficit"
    elif score_gap > 0:
        gap_phrase = f"{score_gap} point lead"
    else:
        gap_phrase = "tied scoreboard"
    current_game = st.session_state.game_number
    max_games = st.session_state.get("max_games", DEFAULT_MAX_GAMES)
    games_remaining = max(0, max_games - current_game + 1)
    prompt = f"""
You are {agent['name']} deciding Game {st.session_state.game_number}.

Scoreboard: you {wins_agent}W/{points_agent}pts, opponent {wins_opponent}W/{points_opponent}pts, draws {draws}.
Series: {current_game}/{max_games} (games left including this one: {games_remaining}). Current gap: {gap_phrase}.

Choosing minimax hands the whole game to a perfect helper.
- Entry fee this game: {next_fee} points (base {fee_info['base_fee']} + adjustment {adjustment_str}). Fee is never refunded.
- Manual win = +10 points. Minimax win = +{10 - next_fee} (because of the fee). Losses/draws = 0.
- You may choose minimax ONLY while trailing in tournament points.

Decide if the guaranteed win is worth the fee: use minimax when you urgently need certainty; otherwise keep the full 10-point upside by playing yourself.

Return ONLY JSON:
{{"use_minimax": true/false}}
"""
    if not client:
        return False, "Set OPENAI_API_KEY to allow minimax decisions.", next_fee

    try:
        response = client.chat.completions.create(
            model=agent["model"],
            messages=[
                {
                    "role": "system",
                    "content": "Decide whether to delegate the entire game to a flawless minimax helper.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
    except Exception as exc:  # pragma: no cover - network call
        return False, f"{agent['name']} minimax decision error: {exc}", next_fee

    content = response.choices[0].message.content.strip()
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        return False, f"{agent['name']} returned invalid minimax decision.", next_fee
    payload = json.loads(content[start : end + 1])
    return bool(payload.get("use_minimax", False)), "", next_fee


def ensure_pregame_minimax_choices() -> None:
    if st.session_state.pregame_decided:
        return
    if not st.session_state.allow_minimax:
        st.session_state.minimax_flags = {"A": False, "B": False}
        st.session_state.minimax_fee_paid = {"A": 0, "B": 0}
        st.session_state.llm_status = (
            "Minimax assistance is disabled for this game. Both agents will play manually."
        )
        st.session_state.pregame_decided = True
        return
    messages: List[str] = []
    for agent_key in ("A", "B"):
        use_minimax, warning, next_fee = llm_decide_minimax(agent_key)
        fee_info = minimax_fee_details(agent_key)
        adjustment_str = (
            f"{fee_info['adjustment']:+d}"
            if isinstance(fee_info["adjustment"], int)
            else str(fee_info["adjustment"])
        )
        if warning:
            st.session_state.minimax_flags[agent_key] = False
            st.session_state.minimax_fee_paid[agent_key] = 0
            messages.append(
                f"{AGENTS[agent_key]['name']} decision issue ({warning}) — defaulting to manual play."
            )
            continue
        points_agent = st.session_state.tournament_points.get(agent_key, 0)
        opponent_key = "B" if agent_key == "A" else "A"
        points_opponent = st.session_state.tournament_points.get(opponent_key, 0)
        if points_agent >= points_opponent:
            st.session_state.minimax_flags[agent_key] = False
            st.session_state.minimax_fee_paid[agent_key] = 0
            messages.append(
                f"{AGENTS[agent_key]['name']} is not trailing on points, so minimax is disabled for this game."
            )
            continue
        st.session_state.minimax_flags[agent_key] = use_minimax
        if use_minimax and points_agent >= next_fee:
            st.session_state.tournament_points[agent_key] -= next_fee
            st.session_state.minimax_fee_paid[agent_key] = next_fee
            st.session_state.minimax_usage_count[agent_key] = (
                st.session_state.minimax_usage_count.get(agent_key, 0) + 1
            )
            messages.append(
                f"{AGENTS[agent_key]['name']} locked in minimax for Game "
                f"{st.session_state.game_number} (entry fee −{next_fee}; base {fee_info['base_fee']}, score gap {fee_info['score_gap']}, adj {adjustment_str})."
            )
        elif use_minimax and points_agent < next_fee:
            st.session_state.minimax_flags[agent_key] = False
            st.session_state.minimax_fee_paid[agent_key] = 0
            messages.append(
                f"{AGENTS[agent_key]['name']} wanted minimax but needs {next_fee} points and only has "
                f"{points_agent} (base {fee_info['base_fee']}, score gap {fee_info['score_gap']}, adj {adjustment_str}), so will play manually."
            )
        else:
            st.session_state.minimax_fee_paid[agent_key] = 0
            messages.append(
                f"{AGENTS[agent_key]['name']} will play manually in Game {st.session_state.game_number}."
            )
    st.session_state.llm_status = " ".join(messages)
    st.session_state.pregame_decided = True


def execute_agent_move(agent_key: str) -> bool:
    if "human_pending_move" not in st.session_state:
        st.session_state.human_pending_move = {"A": None, "B": None}
    if "human_agents" not in st.session_state:
        st.session_state.human_agents = {"A": False, "B": False}
    agent = AGENTS[agent_key]
    opponent_key = "B" if agent_key == "A" else "A"
    symbol = agent["symbol"]
    opponent_symbol = AGENTS[opponent_key]["symbol"]

    if st.session_state.human_agents.get(agent_key):
        pending_move = st.session_state.human_pending_move.get(agent_key)
        if pending_move is None:
            st.session_state.llm_status = (
                f"{agent['name']} (human) to move — select a cell to continue."
            )
            return False
        row, col = pending_move
        idx = row * 3 + col
        st.session_state.human_pending_move[agent_key] = None
        if st.session_state.board[idx]:
            st.session_state.llm_status = (
                f"{agent['name']} selected an occupied square; please choose again."
            )
            return False
        st.session_state.board[idx] = symbol
        log_move(agent_key, row, col, "Human")
        st.session_state.llm_status = (
            f"{agent['name']} (human) played at row {row + 1}, col {col + 1}."
        )
        return True

    if st.session_state.minimax_flags.get(agent_key):
        _, move_idx = minimax(
            st.session_state.board[:],
            symbol,
            symbol,
            opponent_symbol,
        )
        if move_idx is None:
            st.session_state.llm_status = f"{agent['name']} minimax found no move."
            st.session_state.running = False
            return False
        st.session_state.board[move_idx] = symbol
        row, col = divmod(move_idx, 3)
        log_move(agent_key, row, col, "Minimax")
        st.session_state.llm_status = (
            f"{agent['name']} is executing its pre-game minimax plan "
            f"and moved to ({row + 1}, {col + 1})."
        )
        return True

    move: Optional[Tuple[int, int]] = None
    last_error: Optional[str] = None
    for attempt in range(1, MAX_LLM_RETRIES + 1):
        move, error, tokens_used = llm_pick_move(
            st.session_state.board[:],
            symbol,
            opponent_symbol,
            agent["name"],
            agent["model"],
        )
        st.session_state.current_tokens[agent_key] += tokens_used
        if error:
            last_error = error or "unknown LLM issue"
            st.session_state.llm_status = (
                f"{agent['name']} retry {attempt}/{MAX_LLM_RETRIES} failed: {last_error}"
            )
            move = None
            continue
        last_error = None
        break

    if move is None:
        reason = last_error or "could not produce a legal move"
        return restart_game_after_invalid_move(agent_key, opponent_key, reason)

    row, col = move
    idx = row * 3 + col
    st.session_state.board[idx] = symbol
    log_move(agent_key, row, col, "LLM")
    st.session_state.llm_status = (
        f"{agent['name']} played via LLM at row {row + 1}, col {col + 1}."
    )
    return True


def refund_current_minimax_fees() -> None:
    for agent_key in ("A", "B"):
        paid = st.session_state.minimax_fee_paid.get(agent_key, 0)
        if paid:
            st.session_state.tournament_points[agent_key] += paid
            st.session_state.minimax_fee_paid[agent_key] = 0


def restart_game_after_invalid_move(
    agent_key: str, opponent_key: str, reason: str
) -> bool:
    retries = st.session_state.get("game_retry_count", 0)
    if retries >= MAX_GAME_RETRIES:
        st.session_state.llm_status = (
            f"{AGENTS[agent_key]['name']} repeatedly produced invalid moves "
            f"({reason}). Game forfeited to {AGENTS[opponent_key]['name']}."
        )
        log_game_incident(
            st.session_state.game_number,
            f"{AGENTS[agent_key]['name']} forfeited after invalid moves ({reason}).",
        )
        finish_game(AGENTS[opponent_key]["symbol"])
        st.rerun()
        return False

    retries += 1
    st.session_state.game_retry_count = retries
    refund_current_minimax_fees()
    log_game_incident(
        st.session_state.game_number,
        f"{AGENTS[agent_key]['name']} invalid move ({reason}); restart #{retries}.",
    )
    st.session_state.llm_status = (
        f"{AGENTS[agent_key]['name']} produced an invalid move ({reason}). "
        f"Restarting Game {st.session_state.game_number} "
        f"(retry {retries}/{MAX_GAME_RETRIES})."
    )
    reset_board()
    st.session_state.pregame_decided = False
    st.session_state.opening_move_done = False
    st.session_state.running = True
    st.rerun()
    return False


def finish_game(winner_symbol: Optional[str]) -> None:
    starter = st.session_state.starting_player
    current_game = st.session_state.game_number
    st.session_state.game_retry_count = 0
    if winner_symbol == "Draw":
        winner = "Draw"
    elif winner_symbol == AGENTS["A"]["symbol"]:
        winner = "A"
    elif winner_symbol == AGENTS["B"]["symbol"]:
        winner = "B"
    else:
        winner = "Draw"

    if winner in ("A", "B"):
        st.session_state.winning_line = get_winning_line(st.session_state.board)
    else:
        st.session_state.winning_line = None
    if winner == "Draw":
        result_message = f"Game {current_game} ended in a draw."
    else:
        winner_name = AGENTS[winner]["name"]
        winner_symbol_str = AGENTS[winner]["symbol"]
        result_message = (
            f"Game {current_game} won by {winner_name} ({winner_symbol_str})."
        )

    st.session_state.last_finished_board = st.session_state.board[:]
    st.session_state.last_finished_line = st.session_state.winning_line
    st.session_state.last_finished_game = current_game

    if winner == "A":
        st.session_state.tournament_points["A"] += 10
    elif winner == "B":
        st.session_state.tournament_points["B"] += 10

    incidents = st.session_state.game_incidents.pop(current_game, [])

    for agent_key in ("A", "B"):
        if st.session_state.minimax_fee_paid.get(agent_key):
            st.session_state.minimax_fee_paid[agent_key] = 0

    entry = {
        "game": current_game,
        "winner": winner,
        "starter": starter,
        "move_count": len(st.session_state.moves_this_game),
        "A_used_minimax": st.session_state.minimax_flags["A"],
        "B_used_minimax": st.session_state.minimax_flags["B"],
        "A_tokens": st.session_state.current_tokens["A"],
        "B_tokens": st.session_state.current_tokens["B"],
        "A_points": st.session_state.tournament_points["A"],
        "B_points": st.session_state.tournament_points["B"],
        "incidents": incidents,
    }
    st.session_state.scoreboard.append(entry)
    st.session_state.move_history.append(
        {
            "game": current_game,
            "moves": [move.copy() for move in st.session_state.moves_this_game],
        }
    )
    st.session_state.game_number += 1
    st.session_state.starting_player = "B" if st.session_state.starting_player == "A" else "A"

    max_games = st.session_state.get("max_games", DEFAULT_MAX_GAMES)
    if st.session_state.game_number > max_games:
        st.session_state.running = False
        st.session_state.llm_status = f"{result_message} Series complete."
        if not st.session_state.get("series_logged"):
            append_series_log(st.session_state.scoreboard)
            st.session_state.series_logged = True
    else:
        st.session_state.llm_status = (
            f"{result_message} Next up: Game {st.session_state.game_number}."
        )
        reset_board()


def maybe_run_series() -> None:
    if not st.session_state.running:
        return
    max_games = st.session_state.get("max_games", DEFAULT_MAX_GAMES)
    if st.session_state.game_number > max_games:
        st.session_state.running = False
        return

    if not st.session_state.pregame_decided:
        ensure_pregame_minimax_choices()
        if not st.session_state.running:
            return

    if not st.session_state.opening_move_done:
        seed_opening_move()
        st.rerun()
        return

    winner = check_winner(st.session_state.board)
    if winner:
        finish_game(winner)
        st.rerun()
        return

    move_made = execute_agent_move(st.session_state.current_player)
    if not move_made:
        return

    winner = check_winner(st.session_state.board)
    if winner:
        finish_game(winner)
        st.rerun()
        return

    st.session_state.current_player = "B" if st.session_state.current_player == "A" else "A"
    st.rerun()


ensure_state()

model_choices = (
    st.session_state.available_models or load_model_library()
)
for extra in (
    DEFAULT_LLM_A_MODEL,
    DEFAULT_LLM_B_MODEL,
    st.session_state.model_a_choice,
    st.session_state.model_b_choice,
):
    if extra and extra not in model_choices:
        model_choices.append(extra)

def model_format(value: str) -> str:
    return f"{_friendly_label(value)} ({value})"

with st.sidebar:
    st.header("🤖 Arena setup")
    if st.button("🔄 Refresh model catalog"):
        model_choices = load_model_library()
    index_a = model_choices.index(st.session_state.model_a_choice)
    index_b = model_choices.index(st.session_state.model_b_choice)
    selected_model_a = st.selectbox(
        "Model for X (first player)",
        model_choices,
        index=index_a,
        format_func=model_format,
    )
    selected_model_b = st.selectbox(
        "Model for O (second player)",
        model_choices,
        index=index_b,
        format_func=model_format,
    )
    st.session_state.model_a_choice = selected_model_a
    st.session_state.model_b_choice = selected_model_b
    st.session_state.allow_minimax = st.checkbox(
        "Allow minimax helper",
        value=st.session_state.allow_minimax,
        help=(
            "When enabled, each agent may pay a fee to delegate the entire game to minimax. "
            "Fees stay between 1 and 10 points and drop as the current tournament deficit grows."
        ),
    )
    series_options = [5, 10, 20, 30, 50, 100]
    current_max = st.session_state.get("max_games", DEFAULT_MAX_GAMES)
    if current_max not in series_options:
        series_options.append(current_max)
        series_options = sorted(series_options)
    selected_series_length = st.selectbox(
        "Games per series",
        series_options,
        index=series_options.index(current_max),
        help="Choose how many games to run before the series auto-stops.",
    )
    st.session_state.max_games = selected_series_length
    st.session_state.human_agents["A"] = st.checkbox(
        "Play X manually",
        value=st.session_state.human_agents["A"],
        help="When enabled, you will choose moves for the X player.",
    )
    st.session_state.human_agents["B"] = st.checkbox(
        "Play O manually",
        value=st.session_state.human_agents["B"],
        help="When enabled, you will choose moves for the O player.",
    )

LLM_A_MODEL = st.session_state.model_a_choice
LLM_B_MODEL = st.session_state.model_b_choice
LLM_A_LABEL = _friendly_label(LLM_A_MODEL)
LLM_B_LABEL = _friendly_label(LLM_B_MODEL)
AGENTS["A"]["name"] = LLM_A_LABEL
AGENTS["A"]["model"] = LLM_A_MODEL
AGENTS["B"]["name"] = LLM_B_LABEL
AGENTS["B"]["model"] = LLM_B_MODEL

st.title(f"🤖 {LLM_A_LABEL} vs {LLM_B_LABEL} — Tic Tac Toe Series")
st.caption(
    "Agents lock in their minimax choice before each game—no mid-match bailouts—and you can toggle that helper in the sidebar. "
    "Run a series (default 20 games, adjustable in the sidebar) featuring GPT-3.5 vs GPT-4o-mini by default (override via TICTAC_LLM_* env vars). "
    "Wins pay +10 points, draws/losses pay 0, and minimax entry fees stay between 1–10 points based on the score gap."
)

col_controls = st.columns([1, 1, 1])
with col_controls[0]:
    if st.button(
        "▶ Start",
        disabled=st.session_state.running,
    ):
        reset_series()
        st.session_state.running = True
        st.rerun()
with col_controls[1]:
    if st.button("⏸ Pause", disabled=not st.session_state.running):
        st.session_state.running = False
with col_controls[2]:
    if st.button("🔁 Reset Series"):
        reset_series()
        st.rerun()

current_max_games = st.session_state.get("max_games", DEFAULT_MAX_GAMES)
st.subheader(
    f"Game {min(st.session_state.game_number, current_max_games)} of {current_max_games} "
    f"({'running' if st.session_state.running else 'idle'})"
)
board_cols = st.columns(2)
with board_cols[0]:
    st.caption("Live board (current game)")
    render_board(st.session_state.board, st.session_state.get("winning_line"))
with board_cols[1]:
    if st.session_state.last_finished_board:
        last_game_num = st.session_state.last_finished_game
        title = (
            f"Previous game (Game {last_game_num}) final board"
            if last_game_num
            else "Previous game final board"
        )
        st.caption(title)
        render_board(
            st.session_state.last_finished_board,
            st.session_state.get("last_finished_line"),
        )
    else:
        st.caption("Previous game board will appear here once a game finishes.")

if st.session_state.get("opening_seed") is not None:
    st.caption(f"Opening seed: {st.session_state.opening_seed}")

st.write(st.session_state.llm_status or "Press start to begin the series.")
if st.session_state.allow_minimax:
    fee_a = minimax_fee_details("A")
    fee_b = minimax_fee_details("B")
    st.caption(
        f"Next minimax entry fees — {AGENTS['A']['name']}: {fee_a['fee']} pts "
        f"(base {fee_a['base_fee']}, score gap {fee_a['score_gap']}, adj {fee_a['adjustment']:+d}); "
        f"{AGENTS['B']['name']}: {fee_b['fee']} pts "
        f"(base {fee_b['base_fee']}, score gap {fee_b['score_gap']}, adj {fee_b['adjustment']:+d})."
    )
else:
    st.caption("Minimax helper disabled — every move will be LLM-driven without perfect-play overrides.")

current_player = st.session_state.current_player
if "human_pending_move" not in st.session_state:
    st.session_state.human_pending_move = {"A": None, "B": None}
if "human_agents" not in st.session_state:
    st.session_state.human_agents = {"A": False, "B": False}
if st.session_state.human_agents.get(current_player):
    player_name = AGENTS[current_player]["name"]
    st.markdown(f"**Human move controls — {player_name} ({AGENTS[current_player]['symbol']})**")
    if st.session_state.running:
        available = [idx for idx, cell in enumerate(st.session_state.board) if not cell]
        if available:
            labels = [
                f"Cell {idx + 1} (row {idx // 3 + 1}, col {idx % 3 + 1})"
                for idx in available
            ]
            select_key = f"human_move_choice_{current_player}_{st.session_state.game_number}"
            selection = st.selectbox(
                "Choose your move",
                list(range(len(available))),
                format_func=lambda i: labels[i],
                key=select_key,
            )
            if st.button(f"Play move as {player_name}"):
                move_idx = available[selection]
                row, col = divmod(move_idx, 3)
                st.session_state.human_pending_move[current_player] = (row, col)
                st.session_state.llm_status = (
                    f"{player_name} queued move at row {row + 1}, col {col + 1}."
                )
                st.rerun()
        else:
            st.info("Board is full. Waiting for the game to resolve.")
    else:
        st.info("Start or resume the series to submit a manual move.")

st.subheader("Live scoreboard")
scoreboard_entries = st.session_state.scoreboard
agent_name_a = AGENTS["A"]["name"]
agent_name_b = AGENTS["B"]["name"]
display_name_a = agent_name_a
display_name_b = agent_name_b
if display_name_a == display_name_b:
    display_name_a = f"{display_name_a} (X)"
    display_name_b = f"{display_name_b} (O)"

if scoreboard_entries:
    wins_a = sum(1 for e in scoreboard_entries if e["winner"] == "A")
    wins_b = sum(1 for e in scoreboard_entries if e["winner"] == "B")
    draws = sum(1 for e in scoreboard_entries if e["winner"] == "Draw")

    def _summary_tile(label: str, value: str, color: str = "#555") -> None:
        st.markdown(
            f"""
            <div style="text-align:center;padding:0.5rem 0;">
                <div style="font-size:0.9rem;color:#666;">{label}</div>
                <div style="font-size:2rem;font-weight:700;color:{color};">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    color_a = "#2ecc71"
    color_b = "#e74c3c"
    if wins_a == wins_b:
        color_a = color_b = "#3498db"

    summary_cols = st.columns(4)
    with summary_cols[0]:
        _summary_tile("Games played", len(scoreboard_entries))
    with summary_cols[1]:
        _summary_tile(f"{display_name_a} wins", wins_a, color=color_a)
    with summary_cols[2]:
        _summary_tile(f"{display_name_b} wins", wins_b, color=color_b)
    with summary_cols[3]:
        _summary_tile("Draws", draws)
    score_progress_cols = st.columns(2)
    with score_progress_cols[0]:
        st.progress(min(1.0, st.session_state.tournament_points["A"] / 200))
        st.caption(
            f"{display_name_a} score: {st.session_state.tournament_points['A']} pts"
        )
    with score_progress_cols[1]:
        st.progress(min(1.0, st.session_state.tournament_points["B"] / 200))
        st.caption(
            f"{display_name_b} score: {st.session_state.tournament_points['B']} pts"
        )
else:
    st.caption("No games completed yet.")

with st.expander("Game-level details", expanded=True):
    if scoreboard_entries:
        helper_status = (
            "disabled — all moves are LLM-driven."
            if not st.session_state.allow_minimax
            else "enabled — agents may pre-pay to delegate a game."
        )
        opening_seed = st.session_state.get("opening_seed")
        st.caption(f"Minimax helper is {helper_status}")
        if opening_seed is not None:
            st.caption(f"Opening sequence seed: {opening_seed}")
        if st.session_state.llm_status:
            st.caption(f"Game commentary: {st.session_state.llm_status}")

        table_rows = []
        for entry in scoreboard_entries:
            winner_cell = "Draw"
            if entry["winner"] != "Draw" and entry["winner"] in AGENTS:
                winner_agent = AGENTS[entry["winner"]]
                winner_cell = (
                    f"{winner_agent['name']} ({winner_agent['model']}) "
                    f"[{winner_agent['symbol']}]"
                )
            starter_cell = (
                f"{AGENTS[entry['starter']]['name']} ({AGENTS[entry['starter']]['symbol']})"
                if entry.get("starter") in AGENTS
                else entry.get("starter", "?")
            )
            incidents_list = entry.get("incidents", [])
            incidents_cell = " / ".join(incidents_list) if incidents_list else "—"
            table_rows.append(
                {
                    "Game": entry["game"],
                    "Winner": winner_cell,
                    "Starter": starter_cell,
                    "Moves to Finish": entry.get("move_count", 0),
                    f"{display_name_a} Minimax?": "Yes" if entry["A_used_minimax"] else "No",
                    f"{display_name_b} Minimax?": "Yes" if entry["B_used_minimax"] else "No",
                    f"{display_name_a} Tokens": entry.get("A_tokens", 0),
                    f"{display_name_b} Tokens": entry.get("B_tokens", 0),
                    f"{display_name_a} Points": entry.get("A_points", 0),
                    f"{display_name_b} Points": entry.get("B_points", 0),
                    "Incidents": incidents_cell,
                }
            )

        table_df = pd.DataFrame(table_rows)
        st.dataframe(table_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No game-level details available yet.")

if (
    scoreboard_entries
    and not st.session_state.running
    and st.session_state.game_number > st.session_state.get("max_games", DEFAULT_MAX_GAMES)
):
    total_games = len(scoreboard_entries)
    minimax_wins_a = sum(
        1 for e in scoreboard_entries if e["winner"] == "A" and e["A_used_minimax"]
    )
    minimax_wins_b = sum(
        1 for e in scoreboard_entries if e["winner"] == "B" and e["B_used_minimax"]
    )
    minimax_usage_a = sum(1 for e in scoreboard_entries if e["A_used_minimax"])
    minimax_usage_b = sum(1 for e in scoreboard_entries if e["B_used_minimax"])
    st.subheader("Series analytics")
    col_a, col_b, col_draw = st.columns(3)
    with col_a:
        st.metric(f"{display_name_a} win %", f"{(wins_a / total_games * 100):.1f}%")
        st.caption(
            f"Wins via minimax: {minimax_wins_a}/{wins_a}"
            if wins_a
            else "Wins via minimax: 0 (no wins)"
        )
    with col_b:
        st.metric(f"{display_name_b} win %", f"{(wins_b / total_games * 100):.1f}%")
        st.caption(
            f"Wins via minimax: {minimax_wins_b}/{wins_b}"
            if wins_b
            else "Wins via minimax: 0 (no wins)"
        )
    with col_draw:
        st.metric("Draw %", f"{(draws / total_games * 100):.1f}%")
        st.caption(f"{draws} stalemates out of {total_games}")

    losses_a = wins_b
    losses_b = wins_a

    non_draw_games = total_games - draws
    if non_draw_games > 0:
        if wins_a > wins_b:
            leading_agent = display_name_a
            leading_model = LLM_A_MODEL
            successes_for_test = wins_a
            success_axis_label = f"{display_name_a} wins (non-draw games)"
        elif wins_b > wins_a:
            leading_agent = display_name_b
            leading_model = LLM_B_MODEL
            successes_for_test = wins_b
            success_axis_label = f"{display_name_b} wins (non-draw games)"
        else:
            leading_agent = None
            leading_model = None
            successes_for_test = wins_a
            success_axis_label = "Wins (non-draw games)"

        p_value = binomial_two_tailed_p(successes_for_test, non_draw_games)
        significance_cols = st.columns(2)
        with significance_cols[0]:
            st.metric(
                "Non-draw games",
                non_draw_games,
                help="Used to evaluate statistical significance of the win split.",
            )
        with significance_cols[1]:
            st.metric("Win-rate p-value", f"{p_value:.3f}")

        if leading_agent:
            msg = (
                f"{leading_agent} ({leading_model}) lead is statistically significant (p={p_value:.3f})"
                if p_value < 0.05
                else f"No statistical significance yet—need more games to confirm {leading_agent}'s edge (p={p_value:.3f})."
            )
        else:
            msg = (
                f"Series currently tied with {non_draw_games} decisive games "
                f"(p={p_value:.3f}); no significance."
            )

        announce = st.success if leading_agent and p_value < 0.05 else st.info
        announce(msg)

        distribution_df = pd.DataFrame(
            {
                "Win_count": list(range(non_draw_games + 1)),
                "Probability": [
                    comb(non_draw_games, k) * (0.5 ** non_draw_games) for k in range(non_draw_games + 1)
                ],
            }
        )
        distribution_df["Observed"] = distribution_df["Win_count"] == successes_for_test
        fig_dist = px.bar(
            distribution_df,
            x="Win_count",
            y="Probability",
            color="Observed",
            color_discrete_map={True: "#2ecc71", False: "#95a5a6"},
            labels={"Win_count": success_axis_label},
            title="Binomial distribution under a fair matchup (p=0.5)",
        )
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    minimax_long = []
    for agent_name, usage, wins_via_minimax in (
        (display_name_a, minimax_usage_a, minimax_wins_a),
        (display_name_b, minimax_usage_b, minimax_wins_b),
    ):
        minimax_long.append(
            {
                "Agent": agent_name,
                "Metric": "Games using minimax",
                "Value": usage,
            }
        )
        minimax_long.append(
            {
                "Agent": agent_name,
                "Metric": "Wins needing minimax",
                "Value": wins_via_minimax,
            }
        )
    minimax_df = pd.DataFrame(minimax_long)
    st.caption("Minimax reliance")
    fig_minimax = px.bar_polar(
        minimax_df,
        r="Value",
        theta="Metric",
        color="Agent",
        color_discrete_sequence=px.colors.qualitative.Pastel2,
        title="How often each agent leaned on minimax",
    )
    fig_minimax.update_traces(opacity=0.85)
    st.plotly_chart(fig_minimax, use_container_width=True)

    cumulative_wins = []
    minimax_marker_map: Dict[int, List[Dict[str, float]]] = {}
    a_total = 0
    b_total = 0
    for entry in scoreboard_entries:
        if entry["winner"] == "A":
            a_total += 1
        elif entry["winner"] == "B":
            b_total += 1
        cumulative_wins.append(
            {
                "Game": entry["game"],
                "Agent": display_name_a,
                "Wins": a_total,
                "Used minimax": entry["A_used_minimax"],
            }
        )
        cumulative_wins.append(
            {
                "Game": entry["game"],
                "Agent": display_name_b,
                "Wins": b_total,
                "Used minimax": entry["B_used_minimax"],
            }
        )
        if entry["A_used_minimax"]:
            minimax_marker_map.setdefault(entry["game"], []).append(
                {"Agent": display_name_a, "Wins": a_total}
            )
        if entry["B_used_minimax"]:
            minimax_marker_map.setdefault(entry["game"], []).append(
                {"Agent": display_name_b, "Wins": b_total}
            )
    animation_df = pd.DataFrame(cumulative_wins)
    st.caption("Cumulative wins over time")
    fig_animation = px.bar(
        animation_df,
        x="Wins",
        y="Agent",
        color="Agent",
        orientation="h",
        animation_frame="Game",
        animation_group="Agent",
        range_x=[0, max(wins_a, wins_b, 1)],
        color_discrete_sequence=px.colors.qualitative.Set1,
        title="Game-by-game win counter (minimax flag in tooltip)",
    )
    fig_animation.update_layout(transition={"duration": 600})
    marker_style = dict(symbol="line-ns-open", size=26, color="#f39c12", line=dict(width=4))
    initial_game = (
        animation_df["Game"].min() if not animation_df.empty else None
    )
    initial_markers = minimax_marker_map.get(initial_game or 0, [])
    marker_trace = go.Scatter(
        x=[m["Wins"] for m in initial_markers],
        y=[m["Agent"] for m in initial_markers],
        mode="markers",
        marker=marker_style,
        name="Minimax marker",
        hoverinfo="skip",
        showlegend=False,
    )
    fig_animation.add_trace(marker_trace)
    for frame in fig_animation.frames:
        game_id = int(frame.name)
        markers = minimax_marker_map.get(game_id, [])
        frame_data = list(frame.data)
        frame_data.append(
            go.Scatter(
                x=[m["Wins"] for m in markers],
                y=[m["Agent"] for m in markers],
                mode="markers",
                marker=marker_style,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        frame.data = tuple(frame_data)
    st.plotly_chart(fig_animation, use_container_width=True)

st.subheader("Move history")
if st.session_state.move_history:
    last_game = st.session_state.move_history[-1]
    st.write(f"Game {last_game['game']} moves:")
    st.table(
        [
            {
                "Agent": AGENTS.get(move["agent"], {}).get("name", move["agent"]),
                "Symbol": move["symbol"],
                "Row": move["row"],
                "Col": move["col"],
                "Source": move["source"],
            }
            for move in last_game["moves"]
        ]
    )
else:
    st.caption("No moves recorded yet.")

with st.expander("How it works"):
    st.write(
        """
        Two OpenAI LLM agents (defaults: `gpt-3.5-turbo` vs `gpt-4o-mini`) take turns playing
        Tic Tac Toe. Before each game they examine the tournament score and decide—via prompt—
        whether to delegate every move to the perfect-play minimax engine (if the sidebar toggle
        allows it). Opting in costs between 1 and 10 tournament points: it's 10 points while tied
        or leading, then drops by 1 for every 10-point deficit (down to 1) so trailing agents get
        an increasingly cheap bailout. No entry fee is ever refunded.
        Fees are never refunded, wins always grant +10 points, and losses/draws pay 0. There are
        no mid-game requests, so the choice is locked at the opening move.
        Press Start to watch twenty games with randomized opening seeds. After each matchup the app
        logs who won, whether minimax was used, token usage, and cumulative standings. If an LLM
        sends malformed JSON or repeats illegal moves we retry a few times; persistent problems
        pause the series so you can intervene. When the series finishes you'll see analytics covering
        wins/losses/draws, point totals, and minimax reliance.
        """
    )

maybe_run_series()
