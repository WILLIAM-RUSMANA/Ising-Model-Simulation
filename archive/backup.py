def renormalize(self, lattice=None):

    if lattice is None:
        lattice = self.lattice

    N = lattice.shape[0]

    if N % 2 != 0:
        raise ValueError("Lattice size must be even.")

    new_N = N // 2

    coarse = np.zeros((new_N, new_N), dtype=int)

    for i in range(new_N):
        for j in range(new_N):

            # Extract 2x2 block
            block = lattice[2*i:2*i+2, 2*j:2*j+2]

            total = np.sum(block)

            # Majority rule
            if total >= 0:
                coarse[i, j] = 1
            else:
                coarse[i, j] = -1

    return coarse