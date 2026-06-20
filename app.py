# -*- coding: utf-8 -*-
"""
================================================================================
ĐỀ TÀI SỐ 5 — AI CHO TRÒ CHƠI CARO BẰNG HEURISTIC  (Bản Streamlit)
================================================================================
Chạy:   streamlit run app.py
Bàn cờ: 15 x 15  |  Người = X  |  Máy = O  |  Thắng khi đủ 5 quân liên tiếp.

Cấu trúc:
  (1) Khởi tạo bàn cờ        - ma trận 15x15 ký tự '.', 'X', 'O'
  (2) Hàm Heuristic (bộ não) - tấn công + phòng thủ + ưu tiên vị trí
  (3) Quyết định nước đi      - chọn ô tổng điểm Heuristic cao nhất
  (4) Debug Mode             - hiển thị điểm Heuristic ngay trên bàn cờ
  + 3 tình huống bắt buộc (chặn / tấn công / chưa tối ưu) + thống kê 5 ván
================================================================================
"""
import copy
import math
import random
import pandas as pd
import streamlit as st

# ----------------------------- HẰNG SỐ -----------------------------
EMPTY, HUMAN, AI = ".", "X", "O"
WIN_LEN = 5
SIZE = 15
DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
PLAYER_WIN, AI_WIN, DRAW, CONTINUE = "NGUOI_THANG", "MAY_THANG", "HOA", "CHUA_KET_THUC"

PATTERN_SCORE = {
    (5, 2): 1e7, (5, 1): 1e7, (5, 0): 1e7,
    (4, 2): 1e6, (4, 1): 1e5,
    (3, 2): 5e4, (3, 1): 1e3,
    (2, 2): 500, (2, 1): 100,
    (1, 2): 10,  (1, 1): 1,
}


# ===================== (1) KHỞI TẠO / TIỆN ÍCH BÀN CỜ =====================
def create_board(n=SIZE):
    return [[EMPTY] * n for _ in range(n)]


def count_line(b, r, c, dr, dc, p):
    n = len(b)
    cnt = 0
    while 0 <= r < n and 0 <= c < n and b[r][c] == p:
        cnt += 1
        r += dr
        c += dc
    return cnt


def is_win(b, p):
    n = len(b)
    for r in range(n):
        for c in range(n):
            if b[r][c] != p:
                continue
            for dr, dc in DIRS:
                pr, pc = r - dr, c - dc
                if 0 <= pr < n and 0 <= pc < n and b[pr][pc] == p:
                    continue
                if count_line(b, r, c, dr, dc, p) >= WIN_LEN:
                    return True
    return False


def is_full(b):
    return all(cell != EMPTY for row in b for cell in row)


def check_state(b):
    if is_win(b, HUMAN):
        return PLAYER_WIN
    if is_win(b, AI):
        return AI_WIN
    if is_full(b):
        return DRAW
    return CONTINUE


# ===================== (2) HÀM HEURISTIC — BỘ NÃO AI =====================
def line_pattern_score(b, r, c, dr, dc, p):
    n = len(b)
    cnt, open_ends = 1, 0
    rr, cc = r + dr, c + dc
    while 0 <= rr < n and 0 <= cc < n and b[rr][cc] == p:
        cnt += 1; rr += dr; cc += dc
    if 0 <= rr < n and 0 <= cc < n and b[rr][cc] == EMPTY:
        open_ends += 1
    rr, cc = r - dr, c - dc
    while 0 <= rr < n and 0 <= cc < n and b[rr][cc] == p:
        cnt += 1; rr -= dr; cc -= dc
    if 0 <= rr < n and 0 <= cc < n and b[rr][cc] == EMPTY:
        open_ends += 1
    return PATTERN_SCORE.get((min(cnt, 5), open_ends), 0)


def move_value(b, r, c, p):
    total = 0
    b[r][c] = p
    for dr, dc in DIRS:
        total += line_pattern_score(b, r, c, dr, dc, p)
    b[r][c] = EMPTY
    return total


def center_bonus(b, r, c):
    n = len(b)
    ce = (n - 1) / 2
    return max(0, n - (abs(r - ce) + abs(c - ce))) * 0.5


def evaluate_move(b, r, c, me, opp, dw=0.95):
    return move_value(b, r, c, me) + dw * move_value(b, r, c, opp) + center_bonus(b, r, c)


def has_neighbor(b, r, c, rad=2):
    n = len(b)
    for dr in range(-rad, rad + 1):
        for dc in range(-rad, rad + 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and b[nr][nc] != EMPTY:
                return True
    return False


def candidate_moves(b, rad=2):
    n = len(b)
    mv = [(r, c) for r in range(n) for c in range(n)
          if b[r][c] == EMPTY and has_neighbor(b, r, c, rad)]
    return mv if mv else [(n // 2, n // 2)]


# ===================== (3) QUYẾT ĐỊNH NƯỚC ĐI =====================
def ai_choose_move(b, me=AI, opp=HUMAN):
    best, bm, scores = -math.inf, None, {}
    for (r, c) in candidate_moves(b):
        s = evaluate_move(b, r, c, me, opp)
        scores[(r, c)] = s
        if s > best:
            best, bm = s, (r, c)
    return bm, best, scores


def classify_move(b, r, c):
    atk, dfd = move_value(b, r, c, AI), move_value(b, r, c, HUMAN)
    if dfd >= 5e4 and dfd >= atk:
        return "CHẶN", atk, dfd
    if atk >= 5e4:
        return "TẤN CÔNG", atk, dfd
    return "thường", atk, dfd


# ===================== MINIMAX + ALPHA-BETA (so sánh) =====================
def board_score(b):
    s = 0
    n = len(b)
    for r in range(n):
        for c in range(n):
            if b[r][c] == AI:
                for dr, dc in DIRS:
                    s += line_pattern_score(b, r, c, dr, dc, AI)
            elif b[r][c] == HUMAN:
                for dr, dc in DIRS:
                    s -= line_pattern_score(b, r, c, dr, dc, HUMAN)
    return s


def minimax(b, depth, alpha, beta, maxing, top_k=6):
    st_ = check_state(b)
    if st_ == AI_WIN:
        return 1e7 + depth, None
    if st_ == PLAYER_WIN:
        return -1e7 - depth, None
    if st_ == DRAW or depth == 0:
        return board_score(b), None
    me, opp = (AI, HUMAN) if maxing else (HUMAN, AI)
    cands = sorted(candidate_moves(b),
                   key=lambda m: evaluate_move(b, m[0], m[1], me, opp), reverse=True)[:top_k]
    bm = cands[0] if cands else None
    if maxing:
        val = -math.inf
        for (r, c) in cands:
            b[r][c] = AI
            v, _ = minimax(b, depth - 1, alpha, beta, False, top_k)
            b[r][c] = EMPTY
            if v > val:
                val, bm = v, (r, c)
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return val, bm
    else:
        val = math.inf
        for (r, c) in cands:
            b[r][c] = HUMAN
            v, _ = minimax(b, depth - 1, alpha, beta, True, top_k)
            b[r][c] = EMPTY
            if v < val:
                val, bm = v, (r, c)
            beta = min(beta, val)
            if alpha >= beta:
                break
        return val, bm


def weak_player_move(b, rnd=0.35):
    cands = candidate_moves(b)
    if random.random() < rnd:
        return random.choice(cands)
    best, bm = -math.inf, None
    for (r, c) in cands:
        s = evaluate_move(b, r, c, HUMAN, AI, dw=0.4)
        if s > best:
            best, bm = s, (r, c)
    return bm


# ===================== TIỆN ÍCH HIỂN THỊ =====================
def fmt(n):
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}k"
    return str(int(n))


# ===================== STREAMLIT STATE =====================
def init_state():
    ss = st.session_state
    if "board" not in ss:
        ss.board = create_board()
        ss.turn = HUMAN
        ss.state = CONTINUE
        ss.last_move = None
        ss.minimax_mark = None
        ss.explain = "Bạn (X) đi trước. Bật Debug để xem AI 'suy nghĩ' qua điểm Heuristic."
        ss.log = []
        ss.stats = []  # list dict


def reset_game():
    ss = st.session_state
    ss.board = create_board()
    ss.turn = HUMAN
    ss.state = CONTINUE
    ss.last_move = None
    ss.minimax_mark = None
    ss.log = []
    ss.explain = "Ván mới! Bạn (X) đi trước."


def do_human_move(r, c):
    ss = st.session_state
    if ss.state != CONTINUE or ss.turn != HUMAN or ss.board[r][c] != EMPTY:
        return
    ss.board[r][c] = HUMAN
    ss.last_move = (r, c)
    ss.minimax_mark = None
    ss.log.append(f"#{len(ss.log)+1} — Người (X) → ({r}, {c})")
    ss.explain = f"Bạn đánh ({r}, {c})."
    ss.state = check_state(ss.board)
    # Nếu KHÔNG tạm dừng -> máy đáp ngay. Nếu tạm dừng -> chờ bấm "Máy đánh nước này"
    if ss.state == CONTINUE and ss.get("auto_ai", True):
        do_ai_move()
       else:
        ss.turn = AI
        ss.explain = f"Bạn đánh ({r}, {c}). Máy đang chờ bạn bấm ▶️ Máy đánh nước này."


def do_ai_move():
    ss = st.session_state
    if ss.state != CONTINUE:
        return
    move, _, _ = ai_choose_move(ss.board)
    r, c = move
    kind, atk, dfd = classify_move(ss.board, r, c)
    ss.board[r][c] = AI
    ss.last_move = (r, c)
    ss.minimax_mark = None
    ss.turn = HUMAN
    ss.log.append(f"#{len(ss.log)+1} — Máy (O) → ({r}, {c}) · {kind} · TC {fmt(atk)}, PT {fmt(dfd)}")
    ss.explain = f"Máy đánh ({r}, {c}) — **{kind}** (điểm tấn công={fmt(atk)}, phòng thủ={fmt(dfd)})."
    ss.state = check_state(ss.board)


def load_situation(kind):
    ss = st.session_state
    ss.board = create_board()
    ss.turn = AI
    ss.state = CONTINUE
    ss.last_move = None
    ss.minimax_mark = None
    ss.log = []
    b = ss.board
    if kind == "block":
        for rc in [(7, 4), (5, 9), (9, 5)]:
            b[rc[0]][rc[1]] = AI
        for rc in [(7, 5), (7, 6), (7, 7), (7, 8)]:
            b[rc[0]][rc[1]] = HUMAN
        ss.explain = ("**TÌNH HUỐNG 1 — MÁY CHẶN ĐỐI THỦ.** Người (X) có 4 quân liền hàng 7, "
                      "chỉ cần (7,9) là thắng → ô (7,9) có điểm **phòng thủ 10M**. "
                      "Bật Debug để xem, rồi bấm **▶️ Máy đánh nước này**.")
    elif kind == "attack":
        for rc in [(7, 6), (7, 7), (7, 8)]:
            b[rc[0]][rc[1]] = AI
        for rc in [(2, 2), (3, 3), (2, 3), (4, 2)]:
            b[rc[0]][rc[1]] = HUMAN
        ss.explain = ("**TÌNH HUỐNG 2 — MÁY TẤN CÔNG.** Máy (O) có chuỗi 3 hở 2 đầu, "
                      "2 đầu (7,5)/(7,9) có điểm **tấn công ~1M** → máy nối thành chuỗi 4. "
                      "Bấm **▶️ Máy đánh nước này**.")
    elif kind == "subopt":
        for rc in [(6, 6), (6, 7), (7, 8)]:
            b[rc[0]][rc[1]] = AI
        for rc in [(5, 6), (6, 5), (7, 6), (7, 7)]:
            b[rc[0]][rc[1]] = HUMAN
        greedy, _, _ = ai_choose_move(b)
        _, deep = minimax(copy.deepcopy(b), 3, -math.inf, math.inf, True)
        ss.minimax_mark = deep
        ss.explain = (f"**TÌNH HUỐNG 3 — NƯỚC CHƯA TỐI ƯU.** Heuristic (tham lam) chọn "
                      f"🔵 **{greedy}**, còn **Minimax sâu 3 tầng** chọn "
                      f"🟢 **{deep}**. → Heuristic chỉ nhìn cục bộ, "
                      f"cần Minimax/Alpha-Beta để nhìn xa hơn. (Nhìn 2 ô đánh dấu trên bàn cờ.)")


def run_five_games():
    ss = st.session_state
    setups = [(15, True), (15, False), (10, True), (15, True), (10, False)]
    for size, x_first in setups:
        b = create_board(size)
        turn = HUMAN if x_first else AI
        state, mv = CONTINUE, 0
        while state == CONTINUE:
            if turn == HUMAN:
                r, c = weak_player_move(b)
                b[r][c] = HUMAN
                turn = AI
            else:
                (r, c), _, _ = ai_choose_move(b)
                b[r][c] = AI
                turn = HUMAN
            mv += 1
            state = check_state(b)
        res = {AI_WIN: "Máy (O) thắng", PLAYER_WIN: "Người (X) thắng", DRAW: "Hòa"}[state]
        ss.stats.append({"Bàn": f"{size}×{size}",
                         "Đi trước": "Người (X)" if x_first else "Máy (O)",
                         "Kết quả": res, "Số nước": mv})
    ss.explain = "✅ Đã chạy 5 ván tự động — xem bảng thống kê bên dưới."


# ===================== GIAO DIỆN =====================
def render_board(debug):
    """
    Vẽ bàn cờ.
      🔵 = ô Heuristic (tham lam) cho điểm cao nhất.
      🟢 = ô Minimax sâu 3 tầng chọn (chỉ ở tình huống 'nước chưa tối ưu').
      Khi bật Debug: ô trống hiển thị luôn ĐIỂM Heuristic.
    """
    ss = st.session_state
    greedy_key = None
    scores = {}
    # Tính nước tham lam + điểm để đánh dấu (luôn tính khi tới lượt máy)
    if ss.state == CONTINUE and ss.turn == AI:
        greedy_key, _, scores = ai_choose_move(ss.board)
    mm_key = ss.minimax_mark  # ô Minimax (nếu có)

    # CSS thu nhỏ nút cho vừa bàn 15x15
    st.markdown("""
    <style>
      div[data-testid="column"]{padding:1px !important;}
      div.stButton>button{
        width:100%;height:34px;padding:0;font-size:12px;font-weight:700;
        border-radius:4px;border:1px solid #c9a874;
      }
      div.stButton>button[kind="primary"]{
        background:#dbeafe !important;color:#1d4ed8 !important;border:2px solid #2563eb !important;
      }
    </style>""", unsafe_allow_html=True)

    n = len(ss.board)
    for r in range(n):
        cols = st.columns(n, gap="small")
        for c in range(n):
            v = ss.board[r][c]
            key = f"cell_{r}_{c}"
            # marker: 🟢 ô Minimax (ưu tiên) / 🔵 ô Heuristic
            marker = ""
            if (r, c) == mm_key:
                marker = "🟢"
            elif (r, c) == greedy_key:
                marker = "🔵"
            # nhãn ô
            if v == HUMAN:
                label = "❌"
            elif v == AI:
                label = "⭕"
            else:
                base = fmt(scores[(r, c)]) if (debug and (r, c) in scores) else "·"
                label = (marker + base) if marker else base
            # viền xanh dương cho ô Heuristic chọn
            btype = "primary" if (r, c) == greedy_key else "secondary"
            if cols[c].button(label, key=key, type=btype,
                              disabled=(ss.state != CONTINUE or v != EMPTY or ss.turn != HUMAN)):
                do_human_move(r, c)
                st.rerun()


def main():
    st.set_page_config(page_title="AI Caro Heuristic", page_icon="⚫", layout="wide")
    init_state()
    ss = st.session_state

    st.title("⚫ AI cho trò chơi Caro bằng Heuristic")
    st.caption("Người (❌) vs Máy (⭕) · bàn 15×15 · thắng khi đủ 5 quân liên tiếp")

    # ---- Sidebar điều khiển ----
    with st.sidebar:
        st.header("🎛️ Điều khiển")
        debug = st.checkbox("🧠 Nhìn thấu suy nghĩ AI (Debug)", value=False)
        paused = st.checkbox("⏸️ Tạm dừng máy (để xem Debug giữa ván)", value=False,
                             help="Bật: sau khi bạn đánh, máy CHỜ bạn bấm '▶️ Máy đánh nước này' để kịp xem điểm Debug.")
        st.session_state.auto_ai = not paused
        st.button("♻️ Ván mới", on_click=reset_game, use_container_width=True)

        st.divider()
        st.subheader("🎬 Tình huống minh hoạ")
        st.button("🛡️ Máy CHẶN đối thủ", on_click=load_situation, args=("block",), use_container_width=True)
        st.button("⚔️ Máy TẤN CÔNG", on_click=load_situation, args=("attack",), use_container_width=True)
        st.button("⚠️ Nước CHƯA tối ưu", on_click=load_situation, args=("subopt",), use_container_width=True)
        if ss.turn == AI and ss.state == CONTINUE:
            if st.button("▶️ Máy đánh nước này", type="primary", use_container_width=True):
                do_ai_move()
                st.rerun()

        st.divider()
        st.subheader("📊 Thực nghiệm")
        st.button("⏱️ Chạy 5 ván tự động", on_click=run_five_games, use_container_width=True)
        if st.button("🗑️ Xoá thống kê", use_container_width=True):
            ss.stats = []

    # ---- Bố cục chính ----
    col_board, col_info = st.columns([3, 2], gap="large")

    with col_board:
        # trạng thái
        if ss.state == AI_WIN:
            st.success("🤖 MÁY (O) THẮNG!")
        elif ss.state == PLAYER_WIN:
            st.success("🏆 NGƯỜI (X) THẮNG!")
        elif ss.state == DRAW:
            st.info("🤝 HÒA!")
        else:
            st.info("Lượt của bạn (X)" if ss.turn == HUMAN else "Lượt của máy (O) — bấm '▶️ Máy đánh nước này'")
        render_board(debug)
        st.caption("🔵 = ô Heuristic (tham lam) chọn · 🟢 = ô Minimax sâu 3 tầng chọn · "
                   "❌ Người · ⭕ Máy · bật Debug để xem điểm số trên ô trống.")

    with col_info:
        st.markdown("#### 💡 Diễn giải")
        st.markdown(ss.explain)

        st.markdown("#### 📜 Nhật ký nước đi")
        if ss.log:
            st.code("\n".join(ss.log[-12:]), language=None)
        else:
            st.caption("Chưa có nước đi.")

        st.markdown("#### 📊 Thống kê")
        if ss.stats:
            df = pd.DataFrame(ss.stats)
            df.index = [f"Ván {i+1}" for i in range(len(df))]
            x = sum(1 for s in ss.stats if s["Kết quả"] == "Người (X) thắng")
            o = sum(1 for s in ss.stats if s["Kết quả"] == "Máy (O) thắng")
            d = sum(1 for s in ss.stats if s["Kết quả"] == "Hòa")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tổng", len(ss.stats))
            m2.metric("X thắng", x)
            m3.metric("O thắng", o)
            m4.metric("Hòa", d)
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("Chưa có dữ liệu — bấm '⏱️ Chạy 5 ván tự động'.")


if __name__ == "__main__":
    main()
