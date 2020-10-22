import random

class TicTacToe:
    def __init__(self, grid=-1, computer='o', player = 'x'):
        if grid == -1:
            # denotes which spots are open top to bottom left to right
            grid = [[0]]*9
        self.update_positions(grid)

        self.computer = computer
        self.player = player

    def __str__(self):
        """
        Allows to see what the computer thinks is occupied

        returns string containing tic tac toe grid
        """
        grid_text = "\n-------------------\n|"
        for i in range(len(self.grid)):
            grid_text = grid_text + '  %s  '%(self.grid[i][-1])

            if i%3 == 2:
                grid_text = grid_text + '|\n-------------------\n|'
            else:
                grid_text = grid_text + '|'
        return grid_text[0:len(grid_text)-1]

    def update_positions(self, grid):
        """Updates stored grid positions"""
        self.grid = grid

    def select_move(self):
        """
        Selects random move
        
        returns index of square to move to
        """
        while True:
            move = random.randint(0,8)
            if self.grid[move][-1] == ' ':
                return move

    def check_rows(self):
        """
        Checks for win in any row
        
        returns -1 if no win is detected or an index (0-8)
        if a win is detected
        """
        for i in range(0, len(self.grid),3):
            if self.grid[i][-1] != ' ' and self.grid[i][-1] == self.grid[i+1][-1] and self.grid[i+1][-1] == self.grid[i+2][-1]:
                return (i, (self.grid[i], self.grid[i+2]))
        return (-1, None)

    def check_cols(self):
        """
        Checks for win in any column
        
        returns -1 if no win is detected or an index (0-8)
        if a win is detected
        """
        for i in range(3):
            if self.grid[i][-1] != ' ' and self.grid[i][-1] == self.grid[i+3][-1] and self.grid[i+3][-1] == self.grid[i+6][-1]:
                return (i, (self.grid[i], self.grid[i+6]))
        return (-1, None)

    def check_diag(self):
        """
        Checks for win in any diagonal
        
        returns -1 if no win is detected or an index (0-8)
        if a win is detected
        """
        if self.grid[4][-1] != ' ':
            if self.grid[0][-1] == self.grid[4][-1] and self.grid[4][-1] == self.grid[8][-1]:
                return (4, (self.grid[0], self.grid[8]))
            elif self.grid[2][-1] == self.grid[4][-1] and self.grid[4][-1] == self.grid[6][-1]:
                return (4, (self.grid[2], self.grid[6]))
        return (-1, None)

    def check_draw(self):
        count = 0
        for pos in self.grid:
            if pos[-1] != ' ':
                count+=1
        
        if count == 9:
            print('No moves left!')
            return True
        return False

    def check_win(self):
        """
        Checks if a player has won

        returns bool
        """
        wins = [self.check_rows(), self.check_cols(), self.check_diag()]
        for case, pos in wins:
            if case != -1:
                print('Game over!')
                if self.grid[case][-1] == self.computer:
                    print('The computer won!')
                    return (True, pos)
                print('The player won!')
                return (True, pos)

        return (self.check_draw(), None)