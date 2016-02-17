from __future__ import division
import random
'''
To do:
    str(type(x))[7:-2] to type(x)[0]
'''

class Connect3D(object):
    DEFAULT_SIZE = 4
    DEFAULT_PLAYERS = 2
    DEFAULT_SHUFFLE_TURNS = 3
    DEFAULT_SHUFFLE_TYPE = 1
    
    def __init__(self, size=None, players=DEFAULT_PLAYERS):
        
        self.size = self.DEFAULT_SIZE if size is None else max(1, size)
        self.num_players = self.DEFAULT_PLAYERS if players is None else max(1, players)
        
        try:
            self._player
        except AttributeError:
            self._player = random.randint(1, self.num_players)

    def __str__(self):
        
        grid_range = range(self.size)
        grid_output = []
        
        k = 0
        for j in grid_range:
            
            row_top = ' ' * (self.size * 2 + 1) + '_' * (self.size * 4)
            if j:
                row_top = '|{}|{}|{}|'.format(row_top[:self.size * 2 - 1],
                                              '_' * (self.size * 2),
                                              '_' * (self.size * 2 - 1))
            grid_output.append(row_top)
            
            for i in grid_range:
                row_display = '{}/{}'.format(' ' * (self.size * 2 - i * 2), 
                                             ''.join(('{}{}{}/'.format('' if self.grid[k + x] > 9 else ' ', 
                                                                       str(self.grid[k + x]).ljust(1), 
                                                                       '' if self.grid[k + x] > 99 else ' ')) 
                                                     for x in grid_range))
                row_bottom = '{}/{}'.format(' ' * (self.size * 2 - i * 2 - 1),
                                            '___/' * self.size)
                if j != grid_range[-1]:
                    row_display += '{}|'.format(' ' * (i * 2))
                    row_bottom += '{}|'.format(' ' * (i * 2 + 1))
                if j:
                    display_char = row_display[self.size * 4 + 1].strip()
                    row_display = '|{}{}{}'.format(row_display[1:self.size * 4 + 1],
                                                   display_char if display_char else '|',
                                                   row_display[self.size * 4 + 2:])
                    row_bottom = '|{}|{}'.format(row_bottom[1:self.size * 4 + 1],
                                                  row_bottom[self.size * 4 + 2:])
                k += self.size
                grid_output += [row_display, row_bottom]
                
        return '\n'.join(grid_output)

    @classmethod
    def load(cls, data):
        """Calculate the number of segments and split the data into the correct sections.
        
        Parameters:
            x (bytearray): Data to split.
                Acceptable inputs:
                    bytearray(grid_data)
                    bytearray(current_player + grid_data + range_data + progress_data)
        
        Output:
            (num_segments, current_player, grid_data, range_data, progress_data)
        """
    
        if not isinstance(data, bytearray):
            raise ValueError("'{}' input must be a 'bytearray'".format(str(type(data))[7:-2]))
            
        data = data.split(':')
        cube_root = pow(len(data[0]), 1/3)
        
        #Only the grid data is input
        if cube_root == round(cube_root):
            data_player = None
            data_players = None
            data_grid = data[0]
            data_range = bytearray(range(len(data)))
            data_progress = bytearray([])
        
        #All the data is input
        else:
            data_player = data[0].pop(0)
            data_players = data[0].pop(0)
            cube_root = pow(len(data[0])/2, 1/3)
            if cube_root != round(cube_root):
                raise ValueError("invalid input length: '{}'".format(len(''.join(map(str, data))) + 2))
                
            data_length = int(pow(cube_root, 3))
            data_grid = data[0][:data_length]
            data_range = data[0][data_length + 1:data_length * 2]
            data_progress = data[1]
        
        #Create new class
        new_instance = cls(size=int(cube_root), players=data_players)
        if data_players is not None:
            new_instance._player = data_players
        new_instance.grid = data_grid
        new_instance.range = data_range
        new_instance.progress = data_progress
        return new_instance

c = Connect3D.load(bytearray(range(27)))
print c

def split_list(x, n):
    """Split a list by n characters."""
    n = int(n)
    return [x[i:i+n] for i in range(0, len(x), n)]
    
    
def join_list(x):
    """Convert nested lists into one single list."""
    return [j for i in x for j in i]
    
    
class SwapGridData(object):
    """Use the size of the grid to calculate how flip it on the X, Y, or Z axis.
    The flips keep the grid intact but change the perspective of the game.
    """
    def __init__(self, grid_data):
        self.grid_data = list(grid_data)
        self.n = calculate_segments(self.grid_data)
        
        self.range = range(self.n)
        self.range_reverse = self.range[::-1]
        self.group_split = self._group_split()
    
    def _group_split(self):
        return split_list(self.grid_data, pow(self.n, 2))
    
    def fx(self):
        """Flip on the X axis."""
        return join_list(x[::-1] for x in split_list(self.grid_data, self.n))
        
    def fy(self):
        """Flip on the Y axis."""
        return join_list(join_list(split_list(x, self.n)[::-1]) for x in self.group_split)
        
    def fz(self):
        """Flip on the Z axis."""
        return join_list(split_list(self.grid_data, pow(self.n, 2))[::-1])
    
    def rx(self, reverse=None):
        """Rotate on the X axis."""
        n_sq = pow(self.n, 2)
        n_start = pow(self.n, 3) - n_sq
        
        if reverse is None:
            reverse = random.randint(0, 1)
        
        if reverse:
            return [self.grid_data[n_start + i + j * self.n - k * n_sq] for i in self.range_reverse for j in self.range for k in self.range_reverse]
        else:
            return [self.grid_data[n_start + i + j * self.n - k * n_sq] for i in self.range for j in self.range for k in self.range]
            
    def ry(self, reverse=None):
        """Rotate on the Y axis."""
        if reverse is None:
            reverse = random.randint(0, 1)
        if reverse:
            return join_list(j[offset:offset + self.n] for offset in [(self.n - i - 1) * self.n for i in self.range] for j in self.group_split)
        else:
            gs_reverse = self.group_split[::-1]
            return join_list(j[offset:offset + self.n] for offset in [i * self.n for i in self.range] for j in gs_reverse)
            
    def rz(self, reverse=None):
        """Rotate on the Z axis."""
        if reverse is None:
            reverse = random.randint(0, 1)
            
        if reverse:
            return [x[j][i] for x in [split_list(x, self.n) for x in self.group_split] for i in self.range_reverse for j in self.range]
        else:
            return [x[j][i] for x in [split_list(x, self.n)[::-1] for x in self.group_split] for i in self.range for j in self.range]
    
    def reverse(self):
        """Reverse the grid."""
        return self.grid_data[::-1]
