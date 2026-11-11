import random

# 0 = up, 1 = right, 2 = down, 3 = left (y grows downwards)
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
N_ACTIONS = 3  # 0 = straight, 1 = turn right, 2 = turn left


class Snake:
    def __init__(self, size=10, seed=None, features="basic"):
        self.size = size
        self.features = features  # "basic" = 11 inputs, "space" = 11 + free space after each move
        self.n_state = 11 if features == "basic" else 14
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        c = self.size // 2
        self.body = [(c, c), (c - 1, c), (c - 2, c)]  # head first
        self.dir = 1
        self.score = 0
        self.since_food = 0
        self.done = False
        self._place_food()
        return self.state()

    def _place_food(self):
        taken = set(self.body)
        free = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in taken]
        self.food = self.rng.choice(free)

    def _blocked(self, p):
        x, y = p
        if x < 0 or y < 0 or x >= self.size or y >= self.size:
            return True
        return p in self.body[:-1]  # the tail moves away, so it is not a danger

    def _ahead(self, direction):
        dx, dy = DIRS[direction]
        hx, hy = self.body[0]
        return (hx + dx, hy + dy)

    def _free_space(self, start):
        """how many cells can still be reached from start without crossing the snake (flood fill)"""
        if self._blocked(start):
            return 0
        blocked = set(self.body[:-1])
        seen = {start}
        stack = [start]
        while stack:
            x, y = stack.pop()
            for dx, dy in DIRS:
                p = (x + dx, y + dy)
                if 0 <= p[0] < self.size and 0 <= p[1] < self.size and p not in blocked and p not in seen:
                    seen.add(p)
                    stack.append(p)
        return len(seen)

    def state(self):
        d = self.dir
        hx, hy = self.body[0]
        fx, fy = self.food
        extra = []
        if self.features == "space":
            # a dead end can look safe for one step, this tells the net how much room is behind each move
            cells = self.size * self.size
            extra = [self._free_space(self._ahead(k % 4)) / cells for k in (d, d + 1, d - 1)]
        return [
            self._blocked(self._ahead(d)),            # danger straight
            self._blocked(self._ahead((d + 1) % 4)),  # danger right
            self._blocked(self._ahead((d - 1) % 4)),  # danger left
            d == 0, d == 1, d == 2, d == 3,           # current direction
            fx < hx, fx > hx, fy < hy, fy > hy,       # food left, right, up, down
        ] + extra

    def step(self, action):
        """returns (state, reward, done)"""
        self.dir = (self.dir + [0, 1, -1][action]) % 4
        head = self._ahead(self.dir)
        self.since_food += 1

        if self._blocked(head):
            self.done = True
            return self.state(), -10.0, True

        self.body.insert(0, head)
        if head == self.food:
            self.score += 1
            self.since_food = 0
            if len(self.body) == self.size * self.size:  # the board is full
                self.done = True
                return self.state(), 10.0, True
            self._place_food()
            return self.state(), 10.0, False

        self.body.pop()
        if self.since_food > 100:  # going around in circles
            self.done = True
            return self.state(), -10.0, True
        return self.state(), 0.0, False