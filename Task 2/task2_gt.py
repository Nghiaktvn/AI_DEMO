import pygame
import sys
import math
import time
from collections import deque

# kích thước màn hình
WIDTH, HEIGHT = 400, 400
SQUARE_SIZE = WIDTH // 3
LINE_WIDTH = 5

# giới hạn độ sâu để tránh cây vô hạn
MAX_DEPTH = 5  

BLACK = (0, 0, 0)

class Player:
    def __init__(self, symbol):
        # symbol = 'X' hoặc 'O'
        self.symbol = symbol


class Board:
    def __init__(self):
        # bảng 3x3
        self.grid = [[None]*3 for _ in range(3)]

        # lưu lịch sử để xử lý luật "chỉ giữ 4 quân"
        self.history_X = deque()
        self.history_O = deque()

    def make_move(self, r, c, p):
        # nếu ô đã có quân -> không hợp lệ
        if self.grid[r][c] is not None:
            return False, None

        self.grid[r][c] = p
        removed = None

        # chọn lịch sử tương ứng
        hist = self.history_X if p == 'X' else self.history_O
        hist.append((r, c))

        # nếu đặt quân thứ 5 -> xóa quân cũ nhất
        if len(hist) > 4:
            removed = hist.popleft()
            self.grid[removed[0]][removed[1]] = None

        return True, removed

    def undo_move(self, r, c, p, removed):
        # hoàn tác nước đi (dùng trong minimax)
        self.grid[r][c] = None

        hist = self.history_X if p == 'X' else self.history_O
        if hist: hist.pop()

        # khôi phục quân đã bị xóa
        if removed:
            hist.appendleft(removed)
            self.grid[removed[0]][removed[1]] = p

    def get_empty(self):
        # lấy tất cả ô trống
        return [(r,c) for r in range(3) for c in range(3) if self.grid[r][c] is None]

    def check_winner(self):
        # kiểm tra thắng
        g = self.grid
        lines = []

        # hàng + cột
        for i in range(3):
            lines.append(g[i])
            lines.append([g[0][i], g[1][i], g[2][i]])

        # đường chéo
        lines += [
            [g[0][0], g[1][1], g[2][2]],
            [g[0][2], g[1][1], g[2][0]]]

        # nếu có 3 ô giống nhau -> thắng
        for line in lines:
            if line[0] and line.count(line[0]) == 3:
                return line[0]
        return None


class AI(Player):
    def __init__(self, symbol):
        super().__init__(symbol)
        self.nodes = 0   # đếm số node đã duyệt
        self.turn = 0    # đếm lượt

    def get_lines(self, b):
        # lấy tất cả line để evaluate
        g = b.grid
        lines = []
        for i in range(3):
            lines.append(g[i])
            lines.append([g[0][i], g[1][i], g[2][i]])
        lines += [
            [g[0][0], g[1][1], g[2][2]],
            [g[0][2], g[1][1], g[2][0]]]
        return lines

    def evaluate(self, b, depth):
        # hàm đánh giá trạng thái
        w = b.check_winner()

        # xác định đối thủ
        opponent = 'X' if self.symbol == 'O' else 'O'

        # nếu AI thắng -> điểm cao
        if w == self.symbol: return 10 - depth

        # nếu thua -> điểm thấp
        if w == opponent: return depth - 10

        # heuristic: 2 quân + 1 ô trống
        score = 0
        for line in self.get_lines(b):
            if line.count(self.symbol) == 2 and line.count(None) == 1:
                score += 5
            if line.count(opponent) == 2 and line.count(None) == 1:
                score -= 5

        return score

    def minimax(self, b, d, a, beta, is_max, use_ab):
        # đếm node để so sánh performance
        self.nodes += 1

        score = self.evaluate(b, d)

        # dừng nếu đạt độ sâu hoặc thắng/thua
        if d == MAX_DEPTH or abs(score) >= 10:
            return score

        moves = b.get_empty()

        if is_max:
            best = -math.inf
            for r,c in moves:
                # thử nước đi
                _, rm = b.make_move(r,c,self.symbol)

                val = self.minimax(b,d+1,a,beta,False,use_ab)

                # undo lại
                b.undo_move(r,c,self.symbol,rm)

                best = max(best, val)

                # alpha-beta pruning
                if use_ab:
                    a = max(a,best)
                    if beta <= a:
                        break

            return best

        else:
            opponent = 'X' if self.symbol == 'O' else 'O'
            best = math.inf

            for r,c in moves:
                _, rm = b.make_move(r,c,opponent)

                val = self.minimax(b,d+1,a,beta,True,use_ab)

                b.undo_move(r,c,opponent,rm)

                best = min(best, val)

                # alpha-beta pruning
                if use_ab:
                    beta = min(beta,best)
                    if beta <= a:
                        break

            return best

    def best_move(self, b):
        self.turn += 1   
        print(f"\n*** TURN {self.turn} ***")  

        # chạy để so sánh alpha-beta vs baseline
        def run(use_ab):
            self.nodes = 0
            start = time.time()

            for r,c in b.get_empty():
                _, rm = b.make_move(r,c,self.symbol)
                self.minimax(b,0,-math.inf,math.inf,False,use_ab)
                b.undo_move(r,c,self.symbol,rm)

            return time.time()-start, self.nodes

        # alpha-beta
        t1,n1 = run(True)

        # minimax thường (baseline)
        t2,n2 = run(False)

        print(f"Turn {self.turn} - Alpha-Beta: {t1:.5f}s | nodes: {n1}")
        print(f"Turn {self.turn} - Baseline: {t2:.5f}s | nodes: {n2}")

        # in hiệu suất cải thiện
        if n2 != 0 and t2 != 0:
            print(f"Nodes reduced: {(n2 - n1)} ({(n2 - n1)/n2*100:.2f}%)")
            print(f"Time faster: {(t2 - t1):.5f}s ({(t2 - t1)/t2*100:.2f}%)")

        # chọn nước đi tốt nhất (luôn dùng alpha-beta)
        best, move = -math.inf, None
        for r,c in b.get_empty():
            _, rm = b.make_move(r,c,self.symbol)

            val = self.minimax(b,0,-math.inf,math.inf,False,True)

            b.undo_move(r,c,self.symbol,rm)

            if val > best:
                best, move = val, (r,c)

        return move

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT+80))
        pygame.display.set_caption("TicTacToe AI")

        self.board = Board()

        # player = X, AI = O
        self.player = Player('X')
        self.ai = AI('O')

        self.last = None
        self.win = None
        self.over = False

        self.font = pygame.font.SysFont("arial",40,True)
        self.small = pygame.font.SysFont("arial",25)

    def draw(self):
        self.screen.fill((240,240,240))

        # vẽ lưới
        for i in range(1,3):
            pygame.draw.line(self.screen, BLACK,(0,i*SQUARE_SIZE),(WIDTH,i*SQUARE_SIZE),LINE_WIDTH)
            pygame.draw.line(self.screen, BLACK,(i*SQUARE_SIZE,0),(i*SQUARE_SIZE,HEIGHT),LINE_WIDTH)

        # vẽ X O
        for r in range(3):
            for c in range(3):
                v = self.board.grid[r][c]
                if v:
                    color = (200,50,50) if v=='X' else (50,50,200)
                    txt = pygame.font.SysFont("arial",90,True).render(v,True,color)
                    rect = txt.get_rect(center=(c*SQUARE_SIZE+SQUARE_SIZE//2,
                                                r*SQUARE_SIZE+SQUARE_SIZE//2))
                    self.screen.blit(txt,rect)

        # highlight nước đi cuối
        if self.last:
            r,c = self.last
            pygame.draw.rect(self.screen,(255,200,0),
                             (c*SQUARE_SIZE,r*SQUARE_SIZE,SQUARE_SIZE,SQUARE_SIZE),4)

        # hiển thị trạng thái
        pygame.draw.rect(self.screen,(240,240,240),(0,HEIGHT,WIDTH,80))

        if self.win:
            txt = self.font.render(f"{self.win} Wins!",True,(0,150,0))
        else:
            txt = self.small.render("Your turn (X)",True,BLACK)

        self.screen.blit(txt,(20,HEIGHT+20))

        # nút restart
        self.btn = pygame.Rect(WIDTH-160,HEIGHT+20,140,40)
        pygame.draw.rect(self.screen,(100,200,100),self.btn)
        self.screen.blit(self.small.render("Restart",True,BLACK),
                         self.small.render("Restart",True,BLACK).get_rect(center=self.btn.center))

    def reset(self):
        self.board = Board()
        self.ai.turn = 0   
        self.last = None
        self.win = None
        self.over = False

    def run(self):
        while True:
            self.draw()

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

                if e.type == pygame.MOUSEBUTTONDOWN:
                    x,y = e.pos

                    # restart
                    if self.btn.collidepoint(x,y):
                        self.reset(); continue

                    # player đánh
                    if y < HEIGHT and not self.over:
                        r,c = y//SQUARE_SIZE, x//SQUARE_SIZE

                        ok,_ = self.board.make_move(r,c,self.player.symbol)

                        if ok:
                            self.last = (r,c)

                            if self.board.check_winner():
                                self.win=self.player.symbol
                                self.over=True
                            else:
                                # AI đánh
                                m = self.ai.best_move(self.board)
                                if m:
                                    self.board.make_move(m[0],m[1],self.ai.symbol)
                                    self.last = m

                                    if self.board.check_winner():
                                        self.win=self.ai.symbol
                                        self.over=True

            pygame.display.update()

# chạy game
if __name__ == "__main__":
    Game().run()
