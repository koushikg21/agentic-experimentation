import argparse
import csv
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

WINNING_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)

AGENTS: Tuple[Dict[str, str], Dict[str, str]] = (
    {"name": "gpt-5 mini", "model": "gpt-5-mini", "symbol": "X"},
    {"name": "4.1 mini", "model": "gpt-4.1-mini", "symbol": "O"},
)

RESULTS_JSON_FILE = "tournament_results.json"
RESULTS_CSV_FILE = "tournament_results.csv"


def check_winner(board: List[str]) -> Optional[str]:
    """Return 'X', 'O', 'Draw', or None (game continues)."""
    for a, b, c in WINNING_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    if all(cell for cell in board):
        return "Draw"
    return None


def render_board(board: List[str]) -> str:
    """Show a board where empty squares display their index."""
    rows: List[str] = []
    for r in range(3):
        cells = []
        for c in range(3):
            idx = r * 3 + c
            cells.append(board[idx] if board[idx] else str(idx))
        rows.append(" | ".join(cells))
    return "\n---------\n".join(rows)


def available_moves(board: List[str]) -> List[int]:
    return [idx for idx, value in enumerate(board) if not value]


def other_symbol(symbol: str) -> str:
    return "O" if symbol == "X" else "X"


def minimax_best_move(board: List[str], ai_symbol: str) -> int:
    """Return the optimal Tic Tac Toe move for the given symbol."""
    opponent = other_symbol(ai_symbol)

    def minimax(current_symbol: str) -> int:
        winner = check_winner(board)
        if winner:
            if winner == ai_symbol:
                return 1
            if winner == opponent:
                return -1
            return 0
        moves = available_moves(board)
        if current_symbol == ai_symbol:
            best_score = float("-inf")
            for move in moves:
                board[move] = current_symbol
                score = minimax(other_symbol(current_symbol))
                board[move] = ""
                best_score = max(best_score, score)
            return best_score
        best_score = float("inf")
        for move in moves:
            board[move] = current_symbol
            score = minimax(other_symbol(current_symbol))
            board[move] = ""
            best_score = min(best_score, score)
        return best_score

    moves = available_moves(board)
    if not moves:
        return -1
    best_move = moves[0]
    best_score = float("-inf")
    for move in moves:
        board[move] = ai_symbol
        score = minimax(opponent)
        board[move] = ""
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


def minimax_fee(my_points: int, opponent_points: int) -> int:
    """Dynamic minimax cost between 1 and 9 based on point gap."""
    delta = abs(my_points - opponent_points)
    tiers = min(delta // 10, 8)
    return max(1, 9 - tiers)


def parse_move_from_content(content: str) -> Optional[int]:
    """Attempt to extract an integer move from an LLM response."""
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = content[start : end + 1]
        try:
            payload = json.loads(snippet)
            move = payload.get("move")
            if isinstance(move, int):
                return move
            if isinstance(move, str) and move.strip("-").isdigit():
                return int(move)
        except json.JSONDecodeError:
            pass
    digits = re.findall(r"-?\d+", content)
    for token in digits:
        try:
            return int(token)
        except ValueError:
            continue
    return None


def request_llm_move(
    agent: Dict[str, str],
    board: List[str],
    history: List[str],
    memory: List[str],
) -> int:
    """Ask the agent for a move."""
    if not client:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    options = available_moves(board)
    if len(options) == 1:
        return options[0]
    memory_text = "\n".join(memory[-5:]) if memory else "No prior experiences stored."
    prompt = f"""
You are {agent['name']} playing Tic Tac Toe as {agent['symbol']}.

Recent personal memory (oldest first):
{memory_text}

Board indexes (top to bottom):
0 | 1 | 2
3 | 4 | 5
6 | 7 | 8

Current board:
{render_board(board)}

Moves this game:
{os.linesep.join(history) if history else 'None yet'}

Available cells: {options}

Respond ONLY with JSON:
{{"move": <choose one value from {options}>}}

Do not add extra text or comments."""

    for attempt in range(3):
        response = client.chat.completions.create(
            model=agent["model"],
            messages=[
                {
                    "role": "system",
                    "content": "You pick strong, legal Tic Tac Toe moves. Reply using JSON only.",
                },
                {"role": "user", "content": prompt.strip()},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        move = parse_move_from_content(content)
        if move is None:
            continue
        if move not in options:
            continue
        return move
    fallback_move = random.choice(options)
    print(
        f"[fallback] {agent['name']} returned invalid output three times. "
        f"Selecting random legal move: {fallback_move}"
    )
    return fallback_move


def post_game_reflection(
    agent: Dict[str, str],
    opponent: Dict[str, str],
    game_result: Dict[str, Any],
    plan_notes: Dict[str, str],
    minimax_plan: Dict[str, bool],
    points_snapshot: Dict[str, int],
    score_snapshot: Dict[str, int],
) -> str:
    """Ask the agent to reflect on its minimax decision in one sentence."""
    if not client:
        return "Reflection unavailable (no OPENAI_API_KEY)."
    symbol = agent["symbol"]
    used_minimax = minimax_plan.get(symbol, False)
    reason = plan_notes.get(symbol, "")
    winner = game_result["winner"]
    agent_points = points_snapshot.get(agent["name"], 0)
    opponent_points = points_snapshot.get(opponent["name"], 0)
    score_you = score_snapshot.get(agent["name"], 0)
    score_opponent = score_snapshot.get(opponent["name"], 0)
    plan_phrase = "used minimax" if used_minimax else "played manually"
    prompt = f"""
Game {game_result['game']} has ended. Winner: {winner}.
You are {agent['name']} playing as {symbol}. This game you {plan_phrase}.
Your pre-game reason was: {reason or 'none provided'}.
Current tournament points: you {agent_points}, opponent {opponent_points}.
Win scoreboard: you {score_you}, opponent {score_opponent}, draws {score_snapshot.get('Draw', 0)}.

Reflect in ONE sentence on why you chose or avoided minimax and what you learned. Do not exceed one sentence."""
    try:
        response = client.chat.completions.create(
            model=agent["model"],
            messages=[
                {
                    "role": "system",
                    "content": "Provide concise post-game reflections (one sentence).",
                },
                {"role": "user", "content": prompt.strip()},
            ],
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"Reflection error: {exc}"


def decide_minimax_plan(
    agent: Dict[str, str],
    opponent: Dict[str, str],
    win_record: Dict[str, int],
    point_record: Dict[str, int],
    memory: List[str],
    recent_results: List[Dict[str, Any]],
    game_number: int,
    total_games: int,
    minimax_cost: int,
) -> Tuple[bool, str]:
    """Ask the agent whether to use minimax for the entire upcoming game."""
    if not client:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    memory_text = "\n".join(memory[-5:]) if memory else "No prior experiences stored."
    recent_lines = []
    for entry in recent_results[-5:]:
        recent_lines.append(
            f"Game {entry['game']}: winner {entry['winner']} | plan X={entry['minimax_plan']['X']} O={entry['minimax_plan']['O']}"
        )
    recent_text = "\n".join(recent_lines) if recent_lines else "No recent games."
    prompt = f"""
You are {agent['name']} ({agent['symbol']}). Next up is game {game_number}/{total_games}.
Games completed: {game_number - 1}. Games remaining (including this one): {total_games - game_number + 1}.

Total tournament points so far: you {point_record[agent['name']]} pts vs opponent {point_record[opponent['name']]} pts.
Minimax cost this game: {minimax_cost} point(s) (lower when the score gap exceeds 100).
If you win while using minimax, your net gain will be {10 - minimax_cost} points (10 for the win minus the fee). If you lose or draw, you still pay the fee.
The overall tournament winner is determined solely by total points at the end.

Tournament rules:
- Before the game you must decide whether to delegate the ENTIRE game to a perfect minimax solver.
- You may choose minimax only if you currently have enough points to cover the dynamic fee shown above AND you are trailing on points. If scores are tied or you lead, minimax is forbidden.
- Minimax guarantees a win when possible and otherwise forces a draw, but it removes any chance to surprise the opponent manually and immediately costs {minimax_cost} point(s) whether you win or not.
- If you decline minimax, you will play every move yourself for the whole game.

Recent games:
{recent_text}

Your latest memories:
{memory_text}

Return ONLY JSON:
{{"use_minimax": true/false, "reason": "<brief justification>"}}"""
    try:
        response = client.chat.completions.create(
            model=agent["model"],
            messages=[
                {
                    "role": "system",
                    "content": "Decide whether to commit to perfect minimax assistance for the upcoming Tic Tac Toe game.",
                },
                {"role": "user", "content": prompt.strip()},
            ],
        )
    except Exception as exc:
        return False, f"decision error: {exc}"
    content = (response.choices[0].message.content or "").strip()
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            payload = json.loads(content[start : end + 1])
            return bool(payload.get("use_minimax", False)), payload.get("reason", "")
        except json.JSONDecodeError:
            pass
    return False, "invalid response"


def play_single_game(
    game_number: int,
    starting_agent_index: int,
    agent_memories: Dict[str, List[str]],
    minimax_plan: Dict[str, bool],
) -> Dict[str, Any]:
    """Run one game and return details."""
    board = [""] * 9
    move_history: List[str] = []
    current_index = starting_agent_index
    turn = 1
    print("\n" + "=" * 40)
    print(f"Game {game_number} begins. {AGENTS[current_index]['name']} starts as X.")
    while True:
        agent = AGENTS[current_index]
        before_board = render_board(board)
        if minimax_plan[agent["symbol"]]:
            move = minimax_best_move(board, agent["symbol"])
            move_source = "minimax"
        else:
            move = request_llm_move(agent, board, move_history, agent_memories[agent["symbol"]])
            move_source = "manual"
        board[move] = agent["symbol"]
        move_history.append(f"Turn {turn}: {agent['symbol']} -> {move}")
        agent_memories[agent["symbol"]].append(
            f"Game {game_number}, turn {turn}: board before move:\n{before_board}\nChosen cell: {move}"
        )
        print(
            f"Game {game_number} | Turn {turn}: {agent['name']} ({move_source}) "
            f"placed {agent['symbol']} in cell {move}"
        )
        print(render_board(board))
        result = check_winner(board)
        if result:
            if result == "Draw":
                winner_name = "Draw"
                print(f"Game {game_number} ended in a draw.")
            elif result == "X":
                winner_name = AGENTS[0]["name"]
                print(f"Game {game_number} winner: {winner_name}")
            else:
                winner_name = AGENTS[1]["name"]
                print(f"Game {game_number} winner: {winner_name}")
            return {
                "game": str(game_number),
                "winner": winner_name,
                "moves": "; ".join(move_history),
                "minimax_plan": minimax_plan,
            }
        current_index = 1 - current_index
        turn += 1


def run_tournament(game_count: int, allow_minimax: bool = True) -> None:
    """Play a tournament and print the aggregate results."""
    if game_count <= 0:
        raise ValueError("Game count must be positive.")
    scoreboard: Dict[str, int] = {
        AGENTS[0]["name"]: 0,
        AGENTS[1]["name"]: 0,
        "Draw": 0,
    }
    results: List[Dict[str, Any]] = []
    agent_memories: Dict[str, List[str]] = {"X": [], "O": []}
    points: Dict[str, int] = {
        AGENTS[0]["name"]: 0,
        AGENTS[1]["name"]: 0,
    }
    minimax_win_tracker: Dict[str, Dict[str, int]] = {
        AGENTS[0]["name"]: {"with": 0, "without": 0},
        AGENTS[1]["name"]: {"with": 0, "without": 0},
    }
    for game_number in range(1, game_count + 1):
        starting_agent = random.randint(0, 1)
        minimax_plan = {}
        plan_notes = {}
        for agent in AGENTS:
            symbol = agent["symbol"]
            opponent = AGENTS[0] if symbol == "O" else AGENTS[1]
            if not allow_minimax:
                minimax_plan[symbol] = False
                reason = "Minimax disabled via CLI option."
                plan_notes[symbol] = reason
                print(
                    f"Pre-game decision ({agent['name']} as {symbol}): Manual. "
                    f"Reason: {reason}"
                )
                continue
            my_points = points[agent["name"]]
            opponent_points = points[opponent["name"]]
            if my_points >= opponent_points:
                minimax_plan[symbol] = False
                reason = "Forbidden unless trailing; currently tied/leading."
                plan_notes[symbol] = reason
                print(
                    f"Pre-game decision ({agent['name']} as {symbol}): Manual. "
                    f"Reason: {reason}"
                )
                continue
            cost = minimax_fee(my_points, opponent_points)
            if my_points < cost:
                minimax_plan[symbol] = False
                reason = f"Insufficient points (need {cost})."
                plan_notes[symbol] = reason
                print(
                    f"Pre-game decision ({agent['name']} as {symbol}): Manual. "
                    f"Reason: {reason}"
                )
                continue
            use_minimax, reason = decide_minimax_plan(
                agent,
                opponent,
                scoreboard,
                points,
                agent_memories[symbol],
                results,
                game_number,
                game_count,
                cost,
            )
            minimax_plan[symbol] = use_minimax
            plan_notes[symbol] = reason
            choice_label = "Minimax" if use_minimax else "Manual"
            print(
                f"Pre-game decision ({agent['name']} as {symbol}): {choice_label}. "
                f"Reason: {reason or 'none provided'}"
            )
            if use_minimax:
                points[agent["name"]] -= cost
        result = play_single_game(game_number, starting_agent, agent_memories, minimax_plan)
        scoreboard[result["winner"]] += 1
        if result["winner"] in (AGENTS[0]["name"], AGENTS[1]["name"]):
            winner_symbol = "X" if result["winner"] == AGENTS[0]["name"] else "O"
            used_minimax = result["minimax_plan"][winner_symbol]
            key = "with" if used_minimax else "without"
            minimax_win_tracker[result["winner"]][key] += 1
            points[result["winner"]] += 10
        result["plan_notes"] = plan_notes
        result["points_snapshot"] = {
            AGENTS[0]["name"]: points[AGENTS[0]["name"]],
            AGENTS[1]["name"]: points[AGENTS[1]["name"]],
        }
        result["score_snapshot"] = {
            AGENTS[0]["name"]: scoreboard[AGENTS[0]["name"]],
            AGENTS[1]["name"]: scoreboard[AGENTS[1]["name"]],
            "Draw": scoreboard["Draw"],
        }
        reflections: Dict[str, str] = {}
        for agent in AGENTS:
            symbol = agent["symbol"]
            opponent = AGENTS[0] if symbol == "O" else AGENTS[1]
            reflections[symbol] = post_game_reflection(
                agent,
                opponent,
                result,
                plan_notes,
                minimax_plan,
                result["points_snapshot"],
                result["score_snapshot"],
            )
        result["reflections"] = reflections
        results.append(result)
    print("\n" + "=" * 60)
    print("Tournament complete!")
    print(f"Games played: {game_count}")
    for agent in AGENTS:
        print(f"{agent['name']} wins: {scoreboard[agent['name']]}")
    print(f"Draws: {scoreboard['Draw']}")
    print("\nTournament points (win=+10, minimax=-5):")
    for agent in AGENTS:
        print(f"{agent['name']}: {points[agent['name']]} pts")
    print("\nWins using minimax vs manual decisions:")
    for agent in AGENTS:
        stats = minimax_win_tracker[agent["name"]]
        print(
            f"{agent['name']}: {stats['with']} wins with minimax, {stats['without']} wins without minimax"
        )
    print("\nDetailed results:")
    for result in results:
        usage = result["minimax_plan"]
        notes = result.get("plan_notes", {})
        snapshot = result.get("points_snapshot", {})
        score = result.get("score_snapshot", {})
        reflections = result.get("reflections", {})
        print(
            f"Game {result['game']}: {result['winner']} ({result['moves']}) | "
            f"Plan -> X: {usage['X']} ({notes.get('X','')}), O: {usage['O']} ({notes.get('O','')}) | "
            f"Points now: {AGENTS[0]['name']} {snapshot.get(AGENTS[0]['name'], 0)} pts, "
            f"{AGENTS[1]['name']} {snapshot.get(AGENTS[1]['name'], 0)} pts | "
            f"Scoreboard: {AGENTS[0]['name']} {score.get(AGENTS[0]['name'], 0)} wins, "
            f"{AGENTS[1]['name']} {score.get(AGENTS[1]['name'], 0)} wins, Draws {score.get('Draw', 0)}"
        )
        for agent in AGENTS:
            symbol = agent["symbol"]
            print(
                f"  Reflection ({agent['name']} as {symbol}): "
                f"{reflections.get(symbol, 'No reflection recorded.')}"
            )
    suffix = "_manual" if not allow_minimax else ""
    persist_results(results, suffix=suffix)


def persist_results(results: List[Dict[str, Any]], suffix: str = "") -> None:
    """Write tournament results to JSON and CSV for downstream analysis."""
    rows: List[Dict[str, Any]] = []
    agent_x = AGENTS[0]["name"]
    agent_o = AGENTS[1]["name"]
    for result in results:
        points = result.get("points_snapshot", {})
        score = result.get("score_snapshot", {})
        notes = result.get("plan_notes", {})
        reflections = result.get("reflections", {})
        row = {
            "game": int(result["game"]),
            "winner": result["winner"],
            "minimax_plan_x": result["minimax_plan"]["X"],
            "minimax_plan_o": result["minimax_plan"]["O"],
            "reason_x": notes.get("X", ""),
            "reason_o": notes.get("O", ""),
            "reflection_x": reflections.get("X", ""),
            "reflection_o": reflections.get("O", ""),
            "points_x": points.get(agent_x, 0),
            "points_o": points.get(agent_o, 0),
            "score_wins_x": score.get(agent_x, 0),
            "score_wins_o": score.get(agent_o, 0),
            "score_draws": score.get("Draw", 0),
            "points_gap": points.get(agent_o, 0) - points.get(agent_x, 0),
        }
        rows.append(row)

    json_base, json_ext = os.path.splitext(RESULTS_JSON_FILE)
    csv_base, csv_ext = os.path.splitext(RESULTS_CSV_FILE)
    json_path = f"{json_base}{suffix}{json_ext}"
    csv_path = f"{csv_base}{suffix}{csv_ext}"

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(rows, jf, indent=2)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "game",
            "winner",
            "minimax_plan_x",
            "minimax_plan_o",
            "reason_x",
            "reason_o",
            "reflection_x",
            "reflection_o",
            "points_x",
            "points_o",
            "score_wins_x",
            "score_wins_o",
            "score_draws",
            "points_gap",
        ]
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Saved tournament logs to {json_path} and {csv_path} "
        f"({len(rows)} games)."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Tic Tac Toe tournament between GPT-5 mini and GPT-4.1 mini."
    )
    parser.add_argument(
        "--games",
        "-g",
        type=int,
        help="Number of games. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--manual-only",
        action="store_true",
        help="Disable minimax delegation so both agents play every move manually.",
    )
    parser.add_argument(
        "--agent-x-model",
        default=AGENTS[0]["model"],
        help="Model ID to use for the X player.",
    )
    parser.add_argument(
        "--agent-x-name",
        default=AGENTS[0]["name"],
        help="Display name for the X player.",
    )
    parser.add_argument(
        "--agent-o-model",
        default=AGENTS[1]["model"],
        help="Model ID to use for the O player.",
    )
    parser.add_argument(
        "--agent-o-name",
        default=AGENTS[1]["name"],
        help="Display name for the O player.",
    )
    return parser.parse_args()


def prompt_for_game_count() -> int:
    while True:
        user_input = input("How many games should the LLMs play? ").strip()
        try:
            value = int(user_input)
            if value > 0:
                return value
        except ValueError:
            pass
        print("Please enter a positive integer.")


def main() -> None:
    if not client:
        raise RuntimeError("Set OPENAI_API_KEY in your environment before running this script.")
    args = parse_args()
    global AGENTS
    AGENTS = (
        {"name": args.agent_x_name, "model": args.agent_x_model, "symbol": "X"},
        {"name": args.agent_o_name, "model": args.agent_o_model, "symbol": "O"},
    )
    game_count = args.games if args.games is not None else prompt_for_game_count()
    run_tournament(game_count, allow_minimax=not args.manual_only)


if __name__ == "__main__":
    main()
