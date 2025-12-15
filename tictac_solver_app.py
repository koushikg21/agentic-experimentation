import json
import os
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="LLM vs LLM Tic Tac Toe", layout="wide")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_LLM_A_MODEL = os.getenv("TICTAC_LLM_A_MODEL", "gpt-3.5-turbo")
DEFAULT_LLM_B_MODEL = os.getenv("TICTAC_LLM_B_MODEL", "gpt-4o-mini")
FALLBACK_MODEL_LIBRARY = [
    "gpt-4o-mini",
    "gpt-4o",
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
MAX_GAMES = 25
MAX_LLM_RETRIES = 3
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


def check_winner(board: List[str]) -> Optional[str]:
    for a, b, c in WINNING_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
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
    if "available_models" not in st.session_state:
        st.session_state.available_models = []
    if "pregame_decided" not in st.session_state:
        st.session_state.pregame_decided = False
    if "minimax_fee_paid" not in st.session_state:
        st.session_state.minimax_fee_paid = {"A": 0, "B": 0}
    if "minimax_usage_count" not in st.session_state:
        st.session_state.minimax_usage_count = {"A": 0, "B": 0}


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


def reset_series() -> None:
    st.session_state.scoreboard = []
    st.session_state.game_number = 1
    st.session_state.running = False
    st.session_state.starting_player = "A"
    st.session_state.opening_index = 0
    st.session_state.move_history = []
    st.session_state.tournament_points = {"A": 0, "B": 0}
    st.session_state.minimax_usage_count = {"A": 0, "B": 0}
    reset_board()


def load_model_library() -> List[str]:
    def add_unique(items: List[str], model_id: Optional[str]) -> None:
        if model_id and model_id not in items:
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
                if not (
                    model_id.startswith("gpt-")
                    or model_id.startswith("o")
                    or model_id.startswith("text-")
                ):
                    continue
                add_unique(models, model_id)
        except Exception as exc:  # pragma: no cover - network call
            st.warning(f"Could not load OpenAI models dynamically: {exc}")
    st.session_state.available_models = models
    return models


def seed_opening_move() -> None:
    if st.session_state.opening_move_done:
        return
    seq_idx = st.session_state.opening_index % len(OPENING_SEQUENCE)
    board_idx = OPENING_SEQUENCE[seq_idx]
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


def render_board(board: List[str]) -> None:
    html = """
    <style>
    .ttt-wrapper {
        display: flex;
        justify-content: center;
        margin: 1rem 0 1.5rem;
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
    </style>
    <div class="ttt-wrapper">
      <div class="ttt-grid">
    """
    for idx, value in enumerate(board):
        cls = ""
        if value == "X":
            cls = "x"
        elif value == "O":
            cls = "o"
        symbol = value if value else "&nbsp;"
        html += f'<div class="ttt-cell {cls}">{symbol}</div>'
    html += """
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

    prompt = f"""
You are {agent_label} playing Tic Tac Toe as {llm_symbol}. Your opponent uses {opponent_symbol}.

Board rows (top to bottom) use _ for empty cells:
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
            temperature=0,
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
    usage_count = st.session_state.minimax_usage_count.get(agent_key, 0)
    next_fee = 5 * (usage_count + 1)
    prompt = f"""
You are {agent['name']} preparing for Game {st.session_state.game_number}.

Current record: you {wins_agent} wins, opponent {wins_opponent} wins, {draws} draws.
Tournament points: you {points_agent}, opponent {points_opponent}.

Before this game starts you must decide if you want the perfect-play minimax engine to handle ALL of your moves.
- Your next minimax attempt costs {next_fee} points (fees rise each use) and is never refunded, even if it pushes your score negative.
- If you are currently ahead on tournament points, you are not allowed to choose minimax (tie or trailing only).
- Wins always award +10 points (whether you used minimax or not); losses and draws award 0.
- If you play manually you avoid the entry fee but still face the usual win/loss scoring.

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
            temperature=0,
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
    messages: List[str] = []
    for agent_key in ("A", "B"):
        use_minimax, warning, next_fee = llm_decide_minimax(agent_key)
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
        if points_agent > points_opponent:
            st.session_state.minimax_flags[agent_key] = False
            st.session_state.minimax_fee_paid[agent_key] = 0
            messages.append(
                f"{AGENTS[agent_key]['name']} is leading on points, so minimax is disabled for this game."
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
                f"{st.session_state.game_number} (entry fee −{next_fee})."
            )
        elif use_minimax and points_agent < next_fee:
            st.session_state.minimax_flags[agent_key] = False
            st.session_state.minimax_fee_paid[agent_key] = 0
            messages.append(
                f"{AGENTS[agent_key]['name']} wanted minimax but needs {next_fee} points and only has "
                f"{points_agent}, so will play manually."
            )
        else:
            st.session_state.minimax_fee_paid[agent_key] = 0
            messages.append(
                f"{AGENTS[agent_key]['name']} will play manually in Game {st.session_state.game_number}."
            )
    st.session_state.llm_status = " ".join(messages)
    st.session_state.pregame_decided = True


def execute_agent_move(agent_key: str) -> bool:
    agent = AGENTS[agent_key]
    opponent_key = "B" if agent_key == "A" else "A"
    symbol = agent["symbol"]
    opponent_symbol = AGENTS[opponent_key]["symbol"]

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
        st.session_state.llm_status = (
            last_error
            or f"{agent['name']} could not produce a legal move; stopping series."
        )
        st.session_state.llm_status += f" Game forfeited to {AGENTS[opponent_key]['name']}."
        finish_game(AGENTS[opponent_key]["symbol"])
        st.rerun()
        return False

    row, col = move
    idx = row * 3 + col
    st.session_state.board[idx] = symbol
    log_move(agent_key, row, col, "LLM")
    st.session_state.llm_status = (
        f"{agent['name']} played via LLM at row {row + 1}, col {col + 1}."
    )
    return True


def finish_game(winner_symbol: Optional[str]) -> None:
    starter = st.session_state.starting_player
    if winner_symbol == "Draw":
        winner = "Draw"
    elif winner_symbol == AGENTS["A"]["symbol"]:
        winner = "A"
    elif winner_symbol == AGENTS["B"]["symbol"]:
        winner = "B"
    else:
        winner = "Draw"

    if winner == "A":
        st.session_state.tournament_points["A"] += 10
    elif winner == "B":
        st.session_state.tournament_points["B"] += 10

    for agent_key in ("A", "B"):
        if st.session_state.minimax_fee_paid.get(agent_key):
            st.session_state.minimax_fee_paid[agent_key] = 0

    st.session_state.scoreboard.append(
        {
            "game": st.session_state.game_number,
            "winner": winner,
            "starter": starter,
            "A_used_minimax": st.session_state.minimax_flags["A"],
            "B_used_minimax": st.session_state.minimax_flags["B"],
            "A_tokens": st.session_state.current_tokens["A"],
            "B_tokens": st.session_state.current_tokens["B"],
            "A_points": st.session_state.tournament_points["A"],
            "B_points": st.session_state.tournament_points["B"],
        }
    )
    st.session_state.move_history.append(
        {
            "game": st.session_state.game_number,
            "moves": [move.copy() for move in st.session_state.moves_this_game],
        }
    )
    st.session_state.game_number += 1
    st.session_state.starting_player = "B" if st.session_state.starting_player == "A" else "A"

    if st.session_state.game_number > MAX_GAMES:
        st.session_state.running = False
        st.session_state.llm_status = "Series complete."
    else:
        reset_board()


def maybe_run_series() -> None:
    if not st.session_state.running:
        return
    if st.session_state.game_number > MAX_GAMES:
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
    "Before each game the agents decide whether to hand control to the perfect-play minimax helper "
    "for the entire match—no mid-game bailouts. Press start to watch up to twenty-five games with rotating "
    "openings. Defaults pit GPT-3.5-turbo against GPT-4o-mini; override via TICTAC_LLM_A_MODEL / "
    "TICTAC_LLM_B_MODEL. "
    f"Current matchup — {LLM_A_LABEL} ({LLM_A_MODEL}) vs {LLM_B_LABEL} ({LLM_B_MODEL}). "
    "Tournament scoring: win = +10 pts, loss/draw = 0 pts, and each successive minimax use gets pricier "
    "(5, 10, 15… points, never refunded)."
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

st.subheader(
    f"Game {min(st.session_state.game_number, MAX_GAMES)} of {MAX_GAMES} "
    f"({'running' if st.session_state.running else 'idle'})"
)
render_board(st.session_state.board)

st.write(st.session_state.llm_status or "Press start to begin the series.")

st.subheader("Scoreboard")
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
    st.table(
        [
            {
                "Game": entry["game"],
                "Winner": (
                    "Draw"
                    if entry["winner"] == "Draw"
                    else AGENTS[entry["winner"]]["name"]
                ),
                "Starter": (
                    f"{AGENTS[entry['starter']]['name']} ({AGENTS[entry['starter']]['symbol']})"
                    if entry.get("starter") in AGENTS
                    else entry.get("starter", "?")
                ),
                f"{display_name_a} Minimax?": "Yes" if entry["A_used_minimax"] else "No",
                f"{display_name_b} Minimax?": "Yes" if entry["B_used_minimax"] else "No",
                f"{display_name_a} Tokens": entry.get("A_tokens", 0),
                f"{display_name_b} Tokens": entry.get("B_tokens", 0),
                f"{display_name_a} Points": entry.get("A_points", 0),
                f"{display_name_b} Points": entry.get("B_points", 0),
            }
            for entry in scoreboard_entries
        ]
    )
    score_cols = st.columns(2)
    with score_cols[0]:
        st.metric(f"{display_name_a} cumulative score", st.session_state.tournament_points["A"])
    with score_cols[1]:
        st.metric(f"{display_name_b} cumulative score", st.session_state.tournament_points["B"])
    st.caption(
        f"Series tally — {display_name_a}: {wins_a} wins, "
        f"{display_name_b}: {wins_b} wins, Draws: {draws}."
    )
    st.caption(
        f"Cumulative score — {display_name_a}: {st.session_state.tournament_points['A']} pts, "
        f"{display_name_b}: {st.session_state.tournament_points['B']} pts."
    )
else:
    st.caption("No games completed yet.")
    st.caption(
        f"Current score — {display_name_a}: {st.session_state.tournament_points['A']} pts, "
        f"{display_name_b}: {st.session_state.tournament_points['B']} pts."
    )

if (
    scoreboard_entries
    and not st.session_state.running
    and st.session_state.game_number > MAX_GAMES
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
    col_plot_a, col_plot_b = st.columns(2)
    with col_plot_a:
        st.caption(f"{display_name_a} ({LLM_A_MODEL}) outcomes")
        results_a_df = pd.DataFrame(
            {
                "Outcome": ["Wins", "Losses", "Draws"],
                "Count": [wins_a, losses_a, draws],
            }
        )
        fig_a = px.bar(
            results_a_df,
            x="Outcome",
            y="Count",
            color="Outcome",
            text="Count",
            title=f"{display_name_a} results",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_a.update_layout(showlegend=False)
        st.plotly_chart(fig_a, use_container_width=True)

    with col_plot_b:
        st.caption(f"{display_name_b} ({LLM_B_MODEL}) outcomes")
        results_b_df = pd.DataFrame(
            {
                "Outcome": ["Wins", "Losses", "Draws"],
                "Count": [wins_b, losses_b, draws],
            }
        )
        fig_b = px.bar(
            results_b_df,
            x="Outcome",
            y="Count",
            color="Outcome",
            text="Count",
            title=f"{display_name_b} results",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_b.update_layout(showlegend=False)
        st.plotly_chart(fig_b, use_container_width=True)

    minimax_df = pd.DataFrame(
        {
            "Agent": [display_name_a, display_name_b],
            "Minimax Used (games)": [minimax_usage_a, minimax_usage_b],
            "Wins needing Minimax": [minimax_wins_a, minimax_wins_b],
        }
    ).set_index("Agent")
    st.caption("Minimax reliance")
    st.bar_chart(minimax_df, use_container_width=True)

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
        whether to delegate every move to the perfect-play minimax engine. Opting in immediately
        costs 5 points for the first use, 10 for the second, 15 for the third, and so on—fees are
        never refunded—while wins always grant +10 points and losses/draws pay 0. There are no
        mid-game requests, so the choice is locked at the opening move.
        Press Start to watch up to twenty-five games with rotating openings. After each matchup the app
        logs who won, whether minimax was used, token usage, and cumulative standings. If an LLM
        sends malformed JSON or repeats illegal moves we retry a few times; persistent problems
        pause the series so you can intervene. When all games finish you'll see analytics covering
        wins/losses/draws, point totals, and minimax reliance.
        """
    )

maybe_run_series()
