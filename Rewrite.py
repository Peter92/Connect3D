from __future__ import division
from collections import defaultdict
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
        
        #Main parts
        self.range = bytearray(self._range_lg)
        self.grid = bytearray(0 for _ in self.range)
        self.flip = FlipGrid(self)
        self.directions = DirectionCalculation(self)
        self.calculate_score()

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

    def set_grid(self, grid, update_score=True):
        """Set a new grid, used for preview purposes."""
        grid = bytearray(grid)
        if len(grid) != self._size_cubed:
            raise ValueError("grid length must be '{}' not '{}'".format(self._size_cubed, len(grid)))
        self.grid = grid
        if update_score:
            self.calculate_score()
        return self
        
    def calculate_score(self):
        
        self.score = defaultdict(int)
        self._score_hashes = set()
        
        for id in self._range_lg:
            self._point_score(id)
        
        return self
    
    def _point_score(self, id):
        player = self.grid[id]
        if not player:
            return
        
        for movement, invalid in self.directions.reverse_directions:
            count = 1
            list_match = [id]
            
            #Search both directions
            for i in (0, 1):
                point = id
                while point not in invalid[i] and 0 <= point < self._size_cubed:
                    point += movement * (-1 if i else 1)
                    if self.grid[point] == player:
                        count += 1
                        list_match.append(point)
                    else:
                        break
                
            #Add a point if enough matches
            if count == self.size:
                row_hash = hash(tuple(sorted(list_match)))
                if row_hash not in self._score_hashes:
                    self._score_hashes.add(row_hash)
                    self.score[player] += 1
        

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
        
class DirectionCalculation(object):
    """Calculate which directions are possible to move in, based on the 6 directions.
    Any combination is fine, as long as it doesn't go back on itself, hence why X, Y 
    and Z have been given two values each, as opposed to just using six values.
    
    Because the code to calculate score will look in one direction then reverse it, 
    the list then needs to be trimmed down to remove any duplicate directions (eg. 
    up/down and upright/downleft are both duplicates)
    
    The code will output the following results, it is possible to use these instead of the class.
        direction_group = {'Y': 'UD', 'X': 'LR', 'Z': 'FB', ' ': ' '}
        opposite_direction = ('B', 'D', 'DF', 'LDB', 'DB', 'L', 'LUB', 'LUF', 'LF', 'RU', 'LB', 'LDF', 'RD')
    """
    
    def __init__(self, c3d):
        direction_group = {}
        direction_group['X'] = 'LR'
        direction_group['Y'] = 'UD'
        direction_group['Z'] = 'FB'
        direction_group[' '] = ' '
        
        #Come up with all possible directions
        all_directions = set()
        for x in [' ', 'X']:
            for y in [' ', 'Y']:
                for z in [' ', 'Z']:
                    x_directions = list(direction_group[x])
                    y_directions = list(direction_group[y])
                    z_directions = list(direction_group[z])
                    for i in x_directions:
                        for j in y_directions:
                            for k in z_directions:
                                all_directions.add((i+j+k).replace(' ', ''))
        
        #Narrow list down to remove any opposite directions
        self.opposite_direction = all_directions.copy()
        for i in all_directions:
            if i in self.opposite_direction:
                new_direction = ''
                for j in list(i):
                    for k in direction_group.values():
                        if j in k:
                            new_direction += k.replace(j, '')
                self.opposite_direction.remove(new_direction)
    
        self.reverse_directions = []
        for direction in self.opposite_direction:
            
            #Get a list of directions and calculate movement amount
            directions = [list(direction)]
            directions += [[j.replace(i, '') for i in directions[0] for j in direction_group.values() if i in j]]
            direction_movement = sum(c3d.move[j] for j in directions[0])
                            
            #Build list of invalid directions
            invalid_directions = [[c3d.edges[j] for j in directions[k]] for k in (0, 1)]
            invalid_directions = [join_list(j) for j in invalid_directions]
            
            self.reverse_directions.append((direction_movement, invalid_directions))
        
            
            
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

g = [0 for i in range(27)]
g[0] = 3
g[1] = 3
g[2] = 3
g[3] = 1
g[5] = 1
g[4] = 1
g[6] = 1
c = Connect3DGame.load(bytearray(range(27)))
d = Connect3D(3).set_grid(g)
c.core.shuffle()
print d
print d.score
