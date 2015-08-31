import itertools, operator
from collections import defaultdict
import random


def calculate_grid_size(grid_data):
    """Cube root the length of grid_data to find the grid size."""
    return int(round(pow(len(grid_data), 1.0/3.0), 0))

def split_string(x, n):
    """Split a list by n characters."""
    n = int(n)
    return [x[i:i+n] for i in range(0, len(x), n)]
    
def join_list(x):
    """Convert nested lists into one single list."""
    return [j for i in x for j in i]
    

def DrawGrid(grid_data):
    """Use the grid_data to output a grid of the correct size.
    Each value in grid_data must be 1 character or formatting will be wrong.
    
    >>> grid_data = range(8)
    
    >>> DrawGrid(grid_data)
         ________
        / 0 / 1 /|
       /___/___/ |
      / 2 / 3 /  |
     /___/___/   |
    |   |____|___|
    |   / 4 /|5 /
    |  /___/_|_/
    | / 6 / 7|/
    |/___/___|
    
    """
    grid_size = calculate_grid_size(grid_data)
    k = 0
    
    grid_range = range(grid_size)
    grid_output = []
    for j in grid_range:
        
        row_top = ' '*(grid_size*2+1) + '_'*(grid_size*4)
        if j:
            row_top = '|' + row_top[:grid_size*2-1] + '|' + '_'*(grid_size*2) + '|' + '_'*(grid_size*2-1) + '|'
        grid_output.append(row_top)
        
        for i in grid_range:
            row_display = ' '*(grid_size*2-i*2) + '/' + ''.join((' ' + str(grid_data[k+x]).ljust(1) + ' /') for x in grid_range)
            k += grid_size
            row_bottom = ' '*(grid_size*2-i*2-1) + '/' + '___/'*grid_size
            
            if j != grid_range[-1]:
                row_display += ' '*(i*2) + '|'
                row_bottom += ' '*(i*2+1) + '|'
            if j:
                row_display = row_display[:grid_size*4+1] + '|' + row_display[grid_size*4+2:]
                row_bottom = row_bottom[:grid_size*4+1] + '|' + row_bottom[grid_size*4+2:]
                
                row_display = '|' + row_display[1:]
                row_bottom = '|' + row_bottom[1:]
            
            grid_output += [row_display, row_bottom]
            
    print '\n'.join(grid_output)
            
class DirectionCalculation(object):
    """Calculate which directions are possible to move in, based on the 6 directions.
    Any combination is fine, as long as it doesn't go back on itself, hence why X, Y and Z have been given two
    values each, as opposed to just using six values.
    Because the code to calculate score will look in one direction then reverse it, the list then needs to be
    trimmed down to remove any duplicate directions (eg. up/down and upright/downleft are both duplicates)
    
    The code will output the following results, it is possible to use these instead of the class.
        direction_group = {'Y': 'UD', 'X': 'LR', 'Z': 'FB', ' ': ' '}
        opposite_direction = ('B', 'D', 'DF', 'LDB', 'DB', 'L', 'LUB', 'LUF', 'LF', 'RU', 'LB', 'LDF', 'RD')
    """
    
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
    opposite_direction = all_directions.copy()
    for i in all_directions:
        if i in opposite_direction:
            new_direction = ''
            for j in list(i):
                for k in direction_group.values():
                    if j in k:
                        new_direction += k.replace(j, '')
            opposite_direction.remove(new_direction)


class PointConversion(object):
    """Used to convert the cell ID to 3D coordinates or vice versa.
    
    The cell ID is from 0 to grid_size^3, and coordinates are from 1 to grid_size.
    This means an ID of 0 is actually (1,1,1), and 3 would be (4,1,1).
    
               - X -
             __1___2_
        /  1/ 0 / 1 /|
       Y   /___/___/ |
      /  2/ 2 / 3 /  |
         /___/___/   |
        |   |____|___|
     | 1|   / 4 /|5 /
     Z  |  /___/_|_/
     | 2| / 6 / 7|/
        |/___/___|
    
    Parameters:
        grid_size:
            Size of the grid.
            Type: int
        
        i:
            Cell ID or coordinates.
            Type int/tuple/list
    
    Functions:
        to_3d:
            Convert cell ID to 3D coordinate.
        to_int:
            Convert 3D coordinate to cell ID.
    """
    def __init__(self, grid_size, i):
        self.grid_size = grid_size
        self.i = i
        
    def to_3d(self):
        """Convert cell ID to a 3D coordinate.
        
        >>> grid_size = 4
        >>> cell_id = 16
        
        >>> PointConversion(grid_size, cell_id).to_3d()
        (1, 1, 2)
        """
        cell_id = int(self.i)
        z = cell_id / pow(self.grid_size, 2) 
        cell_id %= pow(self.grid_size, 2)
        y = cell_id / self.grid_size
        x = cell_id % self.grid_size
        return tuple(cell_id+1 for cell_id in (x, y, z))
    
    def to_int(self):
        """Convert 3D coordinates to the cell ID.
        
        >>> grid_size = 4
        >>> coordinates = (4,2,3)
        
        >>> PointConversion(grid_size, coordinates).to_int()
        39
        """
        x, y, z = [int(i) for i in self.i]
        if all(i > 0 for i in (x, y, z)):
            return (x-1)*pow(self.grid_size, 0) + (y-1)*pow(self.grid_size, 1) + (z-1)*pow(self.grid_size, 2)
        return None



class SwapGridData(object):
    """Use the size of the grid to calculate how flip it on the X, Y, or Z axis.
    This keeps the grid intact but changes the perspective of the game.
    
    Parameters:
        grid_data:
            2D list of grid cells, amount must be a cube number.
            Type: list/tuple
    """
    def __init__(self, grid_data):
        self.grid_data = list(grid_data)
        self.grid_size = calculate_grid_size(self.grid_data)
    
    def x(self):
        """Flip on the X axis.
        
        >>> SwapGridData(range(8)).x()
        [1, 0, 3, 2, 5, 4, 7, 6]
        >>> DrawGrid(SwapGridData(range(8)).x())
             ________
            / 1 / 0 /|
           /___/___/ |
          / 3 / 2 /  |
         /___/___/   |
        |   |____|___|
        |   / 5 /|4 /
        |  /___/_|_/
        | / 7 / 6|/
        |/___/___|
        """
        return join_list(x[::-1] for x in split_string(self.grid_data, self.grid_size))
        
    def y(self):
        """Flip on the Y axis.
        
        >>> SwapGridData(range(8)).y()
        [2, 3, 0, 1, 6, 7, 4, 5]
        >>> DrawGrid(SwapGridData(range(8)).y())
             ________
            / 2 / 3 /|
           /___/___/ |
          / 0 / 1 /  |
         /___/___/   |
        |   |____|___|
        |   / 6 /|7 /
        |  /___/_|_/
        | / 4 / 5|/
        |/___/___|
        """
        group_split = split_string(self.grid_data, pow(self.grid_size, 2))
        return join_list(join_list(split_string(x, self.grid_size)[::-1]) for x in group_split)
        
    def z(self):
        """Flip on the Z axis.
        
        >>> SwapGridData(range(8)).z()
        [4, 5, 6, 7, 0, 1, 2, 3]
        >>> DrawGrid(SwapGridData(range(8)).z())
             ________
            / 4 / 5 /|
           /___/___/ |
          / 6 / 7 /  |
         /___/___/   |
        |   |____|___|
        |   / 0 /|1 /
        |  /___/_|_/
        | / 2 / 3|/
        |/___/___|
        """
        return join_list(split_string(self.grid_data, pow(self.grid_size, 2))[::-1])
        

class Connect3D(object):
    def __init__(self, grid_size=4, _raw_data=None):
        """Set up the grid and which player goes first.
        
        Parameters:
            grid_size:
                How long each side of the grid should be.
                The game works best with even numbers, 4 is recommended.
                Default: 4
                Type: int
                
            _raw_data:
                Passed in from __repr__, contains the grid data and current player.
                Default: None
                Type: str
                Format: 'joined(grid_data).current_player'
        
        Functions:
            _get_winning_player:
                Return list containing the winning player(s)
            play:
                Play the game with 2 players.
            make_move:
                Update the grid with a new value.
            shuffle:
                Attempt to flip the grid based on chance.
            update_score:
                Recalculate the score.
            show_score:
                Display the score.
            draw:
                Display the grid data.
            reset:
                Empty the grid data.
        """
        
        self.current_player = random.randint(0, 1)
        
        #Read from _raw_data
        if _raw_data is not None:
            split_data = _raw_data.split('.')
            self.grid_data = list(i if i != ' ' else '' for i in split_data[0])
            self.grid_size = calculate_grid_size(self.grid_data)
            if len(self.grid_data) != pow(self.grid_size, 3):
                self.grid_data = self.grid_data[:pow(self.grid_size, 3)]
            
            if len(split_data) > 1:
                self.current_player = int(int(split_data[1]))
        
        #Set up class
        else:
            try:
                self.grid_size = int(grid_size)
            except TypeError:
                raise TypeError('grid_size must be an integer')
            self.grid_data = ['' for i in range(pow(grid_size, 3))]
        
        
        self.grid_size_squared = pow(self.grid_size, 2)
        
        #Calculate the edge numbers for each direction
        self.direction_edges = {}
        self.direction_edges['U'] = range(self.grid_size_squared)
        self.direction_edges['D'] = range(self.grid_size_squared*(self.grid_size-1), self.grid_size_squared*self.grid_size)
        self.direction_edges['R'] = [i*self.grid_size+self.grid_size-1 for i in range(self.grid_size_squared)]
        self.direction_edges['L'] = [i*self.grid_size for i in range(self.grid_size_squared)]
        self.direction_edges['F'] = [i*self.grid_size_squared+j+self.grid_size_squared-self.grid_size for i in range(self.grid_size) for j in range(self.grid_size)]
        self.direction_edges['B'] = [i*self.grid_size_squared+j for i in range(self.grid_size) for j in range(self.grid_size)]
        self.direction_edges[' '] = []
        
        #Calculate the addition needed to move in each direction
        self.direction_maths = {}
        self.direction_maths['D'] = self.grid_size_squared
        self.direction_maths['R'] = 1
        self.direction_maths['F'] = self.grid_size
        self.direction_maths['U'] = -self.direction_maths['D']
        self.direction_maths['L'] = -self.direction_maths['R']
        self.direction_maths['B'] = -self.direction_maths['F']
        self.direction_maths[' '] = 0
        
        
    def __repr__(self):
        """Format the data to allow it to be imported again as a new object."""
        grid_data_joined = ''.join(str(i).ljust(1) for i in self.grid_data)
        return "Connect3D(_raw_data='{}.{}')".format(grid_data_joined, self.current_player)
    
    def _get_winning_player(self):
        """Return a list of the player(s) with the highest points.
        
        >>> C3D = Connect3D()
        >>> C3D.update_score()
        
        When X has a higher score.
        >>> C3D.current_points['X'] = 5
        >>> C3D.current_points['O'] = 1
        >>> C3D._get_winning_player()
        ['X']
        
        When both scores are the same.
        >>> C3D.current_points['O'] = 5
        >>> C3D._get_winning_player()
        ['O', 'X']
        
        When there are no winners.
        >>> C3D = Connect3D()
        >>> C3D.update_score()
        >>> C3D._get_winning_player()
        []
        """
        self.update_score()
        sorted_points = sorted(self.current_points.iteritems(), key=operator.itemgetter(1), reverse=True)
        highest_points = sorted([k for k, v in self.current_points.iteritems() if v == sorted_points[0][1]])
        
        return highest_points
        
        
    def play(self, player_symbols='XO', grid_shuffle_chance=None):
        """Start or continue a game.
        
        Parameters:
            player_symbols:
                The two characters to represent each player.
                Length: 2
                Default: 'XO'
                Type: str/list/tuple/int
            
            grid_shuffle_chance:
                Percentage chance to shuffle the grid after each turn.
                Default: None (use default shuffle chance)
                Type: int/float
        """
        #Check there are two symbols
        if isinstance(player_symbols, (tuple, list)):
            player_symbols = ''.join(player_symbols)
        player_symbols = str(player_symbols)
        same_symbols = len(set(player_symbols)) == 1
        if len(player_symbols) != 2 or same_symbols:
            raise ValueError('two{} symbols are needed'.format(' different' if len(set(player_symbols)) == 1 else ''))
        
        self.current_player = int(not self.current_player)
        
        #Game loop
        while True:
            
            #Switch current player
            self.current_player = int(not self.current_player)
            
            self.update_score()
            self.show_score()
            was_flipped = self.shuffle(chance=grid_shuffle_chance)
            self.draw()
            if was_flipped:
                print "Grid was flipped!"
            
            #Check if no spaces are left
            if '' not in self.grid_data:
                winning_player = self._get_winning_player()
                if len(winning_player) == 1:
                    print 'Player {} won!'.format(winning_player[0])
                else:
                    print 'The game was a draw!'
                    
                #Ask to play again and check if answer is a variant of 'yes' or 'ok'
                print 'Play again?'
                play_again = raw_input().lower()
                if any(i in play_again for i in ('y', 'k')):
                    self.reset()
                else:
                    return
                    break
            
            
            #Player takes a move, function returns True if it updates the grid, otherwise loop again
            print "Player {}'s turn...".format(player_symbols[self.current_player])
            while not self.make_move(player_symbols[self.current_player], raw_input().replace(',', ' ').replace('.', ' ').split()):
                print "Grid cell is not available, try again."
                
                
    def make_move(self, id, *args):
        """Update the grid data with a new move.
        
        Parameters:
            id:
                ID to write to the grid.
                Type: str
            
            args:
                Where to place the ID.
                Can be input as an integer (grid cell number), 3 integers, a tuple or list (3D coordinates)
                Type: int/tuple/list
        
        >>> C3D = Connect3D(2)
        
        >>> C3D.make_move('a', 1)
        True
        >>> C3D.make_move('b', -1)
        False
        >>> C3D.make_move('c', 2, 2, 2)
        True
        >>> C3D.make_move('d', [1, 1, 2])
        True
        >>> C3D.make_move('d', (1, 1, 3))
        False
        
        >>> C3D.grid_data
        ['', 'a', '', '', 'd', '', '', 'c']
        >>> C3D.draw()
             ________
            /   / a /|
           /___/___/ |
          /   /   /  |
         /___/___/   |
        |   |____|___|
        |   / d /|  /
        |  /___/_|_/
        | /   / c|/
        |/___/___|
        """
        
        #Convert points to the grid cell ID
        if len(args) == 1:
            if not str(args[0]).replace('-','').isdigit():
                if len(args[0]) == 1:
                    try:
                        i = int(args[0][0])
                    except ValueError:
                        return False
                else:
                    i = PointConversion(self.grid_size, args[0][0]).to_int()
            else:
                i = int(args[0])
        else:
            i = PointConversion(self.grid_size, tuple(args)).to_int()
        
        #Add to grid if cell is empty
        if 0 <= i <len(self.grid_data) and not self.grid_data[i] and i is not None:
            self.grid_data[i] = id
            return True
        else:
            return False
            
            
            
    def shuffle(self, chance=None, second_chance=None, repeats=None, no_shuffle=[]):
        """Mirror the grid in the X, Y, or Z axis.
        
        Each time one of the directions is flipped, there is a 50% chance of it happening again.
        This means it has the same overall chance to flip, but it is not limited to a single axis.
        
        Parameters:
            chance:
                Percent chance of a flip happening.
                Default: 10
                Type: int/float
            
            second_chance:
                Percent chance of subsequent flips happening after the first.
                Default: 50
                Type: int/float
            
            repeats:
                Number of attempts to flip at the above chance.
                Default: 3
                Type: int
            
            no_shuffle:
                List of directions already flipped so it won't reverse anything.
                Type: list
        """
        #Set defaults
        if chance is None:
            chance = 10
        if second_chance is None:
            second_chance = 50
        if repeats is None:
            repeats = 3
        
        
        #Calculate range of random numbers
        chance = min(100, chance)
        if chance > 0:
            chance = int(round(300/chance))-1
        else:
            chance = 0
            
        #Attempt to flip grid
        for i in range(repeats):
            shuffle_num = random.randint(0, chance)
            if shuffle_num in (0, 1, 2) and shuffle_num not in no_shuffle:
                no_shuffle.append(shuffle_num)
                if shuffle_num == 0:
                    self.grid_data = SwapGridData(self.grid_data).x()
                if shuffle_num == 1:
                    self.grid_data = SwapGridData(self.grid_data).y()
                if shuffle_num == 2:
                    self.grid_data = SwapGridData(self.grid_data).z()
                if self.shuffle(chance=second_chance, no_shuffle=no_shuffle) or not not no_shuffle:
                    return True
                    
                    
                    
    def update_score(self):
        """Recalculate the score.
        
        There are 26 total directions from each point, or 13 lines, calculated in the DirectionCalculation() class.
        For each of the 13 lines, look both ways and count the number of values that match the current player.
        
        This will find any matches from one point, so it's simple to then iterate through every point.
        A hash of each line is stored to avoid duplicates.
        """
        
        try:
            self.grid_data_last_updated
        except AttributeError:
            self.grid_data_last_updated = None
        
        if self.grid_data_last_updated != hash(tuple(self.grid_data)):
        
            #Store hash of grid_data in it's current state to avoid unnecessarily running the code again when there's been no changes
            self.grid_data_last_updated = hash(tuple(self.grid_data))
            
            
            self.current_points = defaultdict(int)
            all_matches = set()
            
            #Loop through each point
            for starting_point in range(len(self.grid_data)):
                
                current_player = self.grid_data[starting_point]
                
                if current_player:
                
                    for i in DirectionCalculation().opposite_direction:
                        
                        #Get a list of directions and calculate movement amount
                        possible_directions = [list(i)]
                        possible_directions += [[j.replace(i, '') for i in possible_directions[0] for j in DirectionCalculation().direction_group.values() if i in j]]
                        direction_movement = sum(self.direction_maths[j] for j in possible_directions[0])
                        
                        #Build list of invalid directions
                        invalid_directions = [[self.direction_edges[j] for j in possible_directions[k]] for k in (0, 1)]
                        invalid_directions = [join_list(j) for j in invalid_directions]
                        
                        num_matches = 1
                        list_match = [starting_point]
                        
                        #Use two loops for the opposite directions
                        for j in (0, 1):
                            
                            current_point = starting_point
                            
                            while current_point not in invalid_directions[j] and 0 < current_point < len(self.grid_data):
                                current_point += direction_movement * int('-'[:j] + '1')
                                if self.grid_data[current_point] == current_player:
                                    num_matches += 1
                                    list_match.append(current_point)
                                else:
                                    break
                        
                        #Add a point if enough matches
                        if num_matches == self.grid_size:
                            
                            list_match = hash(tuple(sorted(list_match)))
                            if list_match not in all_matches:
                                all_matches.add(list_match)
                                self.current_points[current_player] += 1
                          
                          
        
                    
    def show_score(self, digits=False, marker='/'):
        """Print the current points.
        
        Parameters:
            digits:
                If the score should be output as a number, or as individual marks.
                Default: False
                Type: bool
            marker:
                What each point should be displayed as.
                Default: '/'
                Type: str
        
        >>> C3D = Connect3D()
        >>> C3D.update_score()
        >>> C3D.current_points['X'] = 5
        >>> C3D.current_points['O'] = 1
        
        >>> C3D.show_score(False, '/')
        Player X: /////  Player O: /
        >>> C3D.show_score(True)
        Player X: 5  Player O: 1
        """
        self.update_score()
        multiply_value = 1 if digits else marker
        print 'Player X: {x}  Player O: {o}'.format(x=multiply_value*(self.current_points['X']), o=multiply_value*self.current_points['O'])
    
    
    def draw(self):
        """Pass the grid data to the draw function."""
        DrawGrid(self.grid_data)
        
        
    def reset(self):
        """Empty the grid without creating a new Connect3D object."""
        self.grid_data = ['' for i in range(pow(self.grid_size, 3))]
