import numpy as np

class MFWOA:
    """
    Multifactorial Whale Optimization Algorithm (MFWOA)
    Dùng để tối ưu hàm mục tiêu (VD: fuzzy entropy, Dice, IoU, loss function).
    """

    def __init__(self, obj_func, dim, bounds, pop_size=20, max_iter=50, rmp_init=0.3):
        self.obj_func = obj_func   # hàm mục tiêu
        self.dim = dim             # số chiều
        self.bounds = bounds       # (min, max)
        self.pop_size = pop_size   # kích thước quần thể
        self.max_iter = max_iter
        self.rmp = rmp_init        # random mating probability (adaptive)

    def optimize(self):
        # --- Khởi tạo quần thể ---
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([self.obj_func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx].copy()
        best_score = fitness[best_idx]

        # --- Vòng lặp chính ---
        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)  # giảm tuyến tính từ 2 -> 0

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                A = 2 * a * r1 - a
                C = 2 * r2
                p = np.random.rand()

                if p < 0.5:
                    if abs(A) < 1:
                        # bao vây con mồi (exploitation)
                        D = abs(C * best - pop[i])
                        pop[i] = best - A * D
                    else:
                        # khám phá (exploration)
                        rand_idx = np.random.randint(0, self.pop_size)
                        D = abs(C * pop[rand_idx] - pop[i])
                        pop[i] = pop[rand_idx] - A * D
                else:
                    # tấn công xoắn ốc
                    distance = abs(best - pop[i])
                    l = np.random.uniform(-1, 1)
                    b = 1
                    pop[i] = distance * np.exp(b * l) * np.cos(2 * np.pi * l) + best

                # --- Giới hạn biên ---
                pop[i] = np.clip(pop[i], self.bounds[0], self.bounds[1])

            # --- Cập nhật fitness ---
            fitness = np.array([self.obj_func(ind) for ind in pop])
            if fitness.min() < best_score:
                best_idx = np.argmin(fitness)
                best = pop[best_idx].copy()
                best_score = fitness[best_idx]
            else:
                # nếu không cải thiện → điều chỉnh rmp (adaptive)
                self.rmp += 0.1 * np.random.normal()

        return best, best_score
