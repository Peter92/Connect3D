from __future__ import division
import random
import base64

class Connect3D(object):
    """Class for holding the game information.
    Supports numbers up to 255.
    """
    DEFAULT_SIZE = 4
    DEFAULT_SHUFFLE_LEVEL = 1
    
    def __init__(self, size=None, shuffle_level=DEFAULT_SHUFFLE_LEVEL):
        
        self.size = self.DEFAULT_SIZE if size is None else max(1, size)
        self.shuffle_level = self.DEFAULT_SHUFFLE_LEVEL if shuffle_level is None else max(0, min(2, shuffle_level))
        
        #Precalculated parts
        self._size_squared = int(pow(self.size, 2))
        self._size_cubed = int(pow(self.size, 3))
        self._range_sm = range(self.size)
        self._range_sm_r = self._range_sm[::-1]
        self._range_md = range(self._size_squared)
        self._range_lg = range(self._size_cubed)
        
        #Main parts
        self.range = bytearray(self._range_lg)
        self.grid = bytearray(0 for _ in self.range)
        self.flip = FlipGrid(self)
        
        #Calculate the edge numbers for each direction
        self.edges = {'U': list(self._range_md),
                      'D': range(self._size_squared * (self.size - 1), self._size_squared * self.size),
                      'R': [i * self.size + self.size - 1 for i in self._range_md],
                      'L': [i * self.size for i in self._range_md],
                      'F': [i * self._size_squared + self._size_squared + j - self.size
                            for i in self._range_sm for j in self._range_sm],
                      'B': [i * self._size_squared + j for i in self._range_sm for j in self._range_sm],
                      ' ': []}
                      
        #Calculate the addition needed to move in each direction
        self.move = {'U': -self._size_squared,
                     'D': self._size_squared,
                     'L': -1,
                     'R': 1,
                     'F': self.size,
                     'B': -self.size,
                     ' ': 0}

    def __repr__(self):
        output = (bytearray([self._player]) + bytearray([self.num_players])
                  + self.grid + self.range + bytearray([':']) + self.progress)
        return "Connect3DGame.load('{}')".format(base64.b64encode(output))

    def __str__(self):
        """Print the current state of the grid."""
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


    def shuffle(self, level=None):
        """Mirror the grid in the X, Y, or Z axis.
        
        A level of 1 is mirror only.
        A level of 2 includes rotation.
        """
        if level is None:
            level = self.shuffle_level
            
        #Shuffle is disabled
        if not level and isinstance(level, bool):
            return
            
        max_shuffles = level * 3
        shuffles = random.sample(range(max_shuffles), random.randint(0, max_shuffles - 1))
        
        all_flips = (self.flip.fx, self.flip.fy, self.flip.fz, 
                     self.flip.rx, self.flip.ry, self.flip.rz, self.flip.reverse)
        
        for i in shuffles:
            self.grid = all_flips[i](self.grid)
            self.range = all_flips[i](self.range)

    def load_grid(self, grid):
        grid = bytearray(grid)
        if len(grid) != self._size_cubed:
            raise ValueError("grid length must be '{}' not '{}'".format(self._size_cubed, len(grid)))
        self.grid = grid
        

class Connect3DGame(object):
    DEFAULT_PLAYERS = 2
    DEFAULT_SHUFFLE_TURNS = 3
    
    def __init__(self, num_players=DEFAULT_PLAYERS, shuffle_turns=DEFAULT_SHUFFLE_TURNS, size=None):
        self.num_players = self.DEFAULT_PLAYERS if num_players is None else max(1, num_players)
        self.shuffle_turns = self.DEFAULT_SHUFFLE_TURNS if shuffle_turns is None else max(0, shuffle_turns)
        self.core = Connect3D(size)
        
        try:
            self._player
        except AttributeError:
            self._player = random.randint(1, self.num_players)
        
    def next_player(self):
        self._player += 1
        self._player %= self.players
    
    def previous_player(self):
        self._player -= 1
        self._player %= self.players
        
    @classmethod
    def load(cls, data):
        """Calculate the number of segments and split the data into the correct sections.
        
        Parameters:
            x (string): Data to split. 
                Must be a base 64 encoded bytearray.
                Acceptable inputs:
                    bytearray(grid_data)
                    bytearray(current_player + num_players + grid_data + range_data)
        """
        if not isinstance(data, (bytearray, str)):
            raise ValueError("'{}' input must be a 'bytearray'".format(type(data).__name__))
        if isinstance(data, str):
            try:
                data = bytearray(base64.b64decode(data))
                if not data.strip():
                    raise TypeError
            except TypeError:
                raise TypeError("input data is not correctly encoded")
            
        cube_root = pow(len(data), 1/3)
        
        #Only the grid data is input
        if cube_root == round(cube_root):
            data_player = None
            data_players = None
            data_grid = data
            data_range = bytearray(range(len(data)))
        
        #All the data is input
        else:
            data_player = data.pop(0)
            data_players = data.pop(0)
            cube_root = pow(len(data)/2, 1/3)
            if cube_root != round(cube_root):
                raise ValueError("invalid input length: '{}'".format(len(''.join(map(str, data))) + 2))
                
            data_length = int(pow(cube_root, 3))
            data_grid = data[:data_length]
            data_range = data[data_length + 1:data_length * 2]
        
        #Create new class
        new_instance = cls(size=int(cube_root), num_players=data_players)
        if data_players is not None:
            new_instance._player = data_players
        new_instance.core.grid = data_grid
        new_instance.core.range = data_range
        return new_instance
        

def split_list(x, n):
    """Split a list by n characters."""
    n = int(n)
    return [x[i:i+n] for i in range(0, len(x), n)]
    
    
def join_list(x):
    """Convert nested lists into one single list."""
    return [j for i in x for j in i]
    
    
class FlipGrid(object):
    """Use the size of the grid to calculate how flip it on the X, Y, or Z axis.
    The flips keep the grid intact but change the perspective of the game.
    """
    def __init__(self, c3d):
        self.c3d = c3d
    
    def fx(self, data):
        """Flip on the X axis."""
        return join_list(x[::-1] for x in split_list(data, self.c3d.size))
        
    def fy(self, data):
        """Flip on the Y axis."""
        return join_list(join_list(split_list(x, self.c3d.size)[::-1]) 
                                   for x in split_list(data, self.c3d._size_squared))
        
    def fz(self, data):
        """Flip on the Z axis."""
        return join_list(split_list(data, pow(self.c3d.size, 2))[::-1])
    
    def rx(self, data, reverse=None):
        """Rotate on the X axis."""
        if reverse is None:
            reverse = random.randint(0, 1)
        
        start = self.c3d._size_cubed - self.c3d._size_squared
        if reverse:
            return [data[start + i + j * self.c3d.size - k * self.c3d._size_squared] 
                    for i in self.c3d._range_sm_r 
                    for j in self.c3d._range_sm 
                    for k in self.c3d._range_sm_r]
        else:
            return [data[start + i + j * self.c3d.size - k * self.c3d._size_squared] 
                    for i in self.c3d._range_sm 
                    for j in self.c3d._range_sm 
                    for k in self.c3d._range_sm]
            
    def ry(self, data, reverse=None):
        """Rotate on the Y axis."""
        if reverse is None:
            reverse = random.randint(0, 1)
        
        split = split_list(data, self.c3d._size_squared)
        if reverse:
            return join_list(j[offset:offset + self.c3d.size] 
                             for offset in [(self.c3d.size - i - 1) * self.c3d.size 
                                            for i in self.c3d._range_sm]
                             for j in split)
        else:
            split = split[::-1]
            return join_list(j[offset:offset + self.c3d.size] 
                             for offset in [i * self.c3d.size 
                                            for i in self.c3d._range_sm] 
                             for j in split)
            
    def rz(self, data, reverse=None):
        """Rotate on the Z axis."""
        if reverse is None:
            reverse = random.randint(0, 1)
            
        split = split_list(data, self.c3d._size_squared)
        if reverse:
            return [x[j][i] 
                    for x in [split_list(x, self.c3d.size) 
                              for x in split] 
                    for i in self.c3d._range_sm_r for j in self.c3d._range_sm]
        else:
            return [x[j][i] 
                    for x in [split_list(x, self.c3d.size)[::-1] 
                              for x in split] 
                    for i in self.c3d._range_sm for j in self.c3d._range_sm]
    
    def reverse(self, data):
        """Reverse the grid."""
        return data[::-1]
        

c = Connect3DGame.load(bytearray(range(27)))
c.core.shuffle()
print c.core
