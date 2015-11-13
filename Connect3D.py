import itertools
import operator
import random
import time
import pygame
import math
from collections import defaultdict
class Connect3DError(Exception):
    pass

BACKGROUND = (250, 250, 255)
LIGHTBLUE = (86, 190, 255)
LIGHTGREY = (200, 200, 200)
GREY = (128, 128, 128)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
SELECTION = {'Default': [WHITE, LIGHTGREY],
             'Hover': [None, BLACK],
             'Waiting': [None, BLACK],
             'Selected': [GREEN, None]}

def mix_colour(*args):
    mixed_colour = [0, 0, 0]
    num_colours = len(args)
    for colour in range(3):
        mixed_colour[colour] = sum(i[colour] for i in args) / num_colours
    return mixed_colour
    
class Connect3D(object):
    """Class to store and use the Connect3D game data.
    The data is stored in a 1D list, but is converted to a 3D representation for the user.
    """
    player_symbols = 'XO'
    default_segments = 4
    default_shuffle_count = 3
    from_string_separator = '/'
    bot_difficulty_default = 'medium'
    
    def __init__(self, segments=default_segments):
        """Set up the grid and which player goes first.
        
        Parameters:
            segments (int): How long each side of the grid should be.
                The game works best with even numbers, 4 is recommended.
        """
        
        #Set current player
        try:
            self.current_player
        except AttributeError:
            self.current_player = random.randint(0, 1)
        
        #Set up grid
        try:
            self.segments = int(segments)
        except TypeError:
            raise TypeError('segments must be an integer')

        self.ai_message = []

        self.play_data = []
        self.segments_squared = pow(self.segments, 2)
        self.segments_cubed = pow(self.segments, 3)
        self.range_data = range(self.segments_cubed)
        self.grid_data = ['' for i in self.range_data]
        self.update_score()
        
        #Calculate the edge numbers for each direction
        self.direction_edges = {}
        self.direction_edges['U'] = range(self.segments_squared)
        self.direction_edges['D'] = range(self.segments_squared*(self.segments-1), self.segments_squared*self.segments)
        self.direction_edges['R'] = [i*self.segments+self.segments-1 for i in range(self.segments_squared)]
        self.direction_edges['L'] = [i*self.segments for i in range(self.segments_squared)]
        self.direction_edges['F'] = [i*self.segments_squared+j+self.segments_squared-self.segments for i in range(self.segments) for j in range(self.segments)]
        self.direction_edges['B'] = [i*self.segments_squared+j for i in range(self.segments) for j in range(self.segments)]
        self.direction_edges[' '] = []
        
        #Calculate the addition needed to move in each direction
        self.direction_maths = {}
        self.direction_maths['D'] = self.segments_squared
        self.direction_maths['R'] = 1
        self.direction_maths['F'] = self.segments
        self.direction_maths['U'] = -self.direction_maths['D']
        self.direction_maths['L'] = -self.direction_maths['R']
        self.direction_maths['B'] = -self.direction_maths['F']
        self.direction_maths[' '] = 0
        
        
    def __repr__(self):
        """Format the data to allow it to be imported again as a new object."""
        grid_data_joined = ''.join(str(i).ljust(1) for i in self.grid_data)
        longest_number = len(str(pow(self.segments, 3) - 1))
        
        repr_format = '{}{s}{}'.format(grid_data_joined, self.current_player, s=self.from_string_separator)
        
        if self.play_data is not None and self.range_data is not None:
            play_data_joined = ''.join(str(i).zfill(longest_number) for i in self.play_data)
            range_data_joined = ''.join(str(i).zfill(longest_number) for i in self.range_data)
            repr_format += '{s}{}{s}{}'.format(play_data_joined, range_data_joined, s=self.from_string_separator)
            
            
        return "Connect3D.from_string('{}')".format(repr_format)
    
    def __str__(self):
        """Use the grid_data to output a grid of the correct size.
        Each value in grid_data must be 1 character or formatting will be wrong.
        
        >>> grid_data = range(8)
        
        >>> print Connect3D.from_string(''.join(str(i) if i != '' else ' ' for i in grid_data))
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
        k = 0
        
        grid_range = range(self.segments)
        grid_output = []
        
        for j in grid_range:
            
            row_top = ' '*(self.segments*2+1) + '_'*(self.segments*4)
            if j:
                row_top = '|' + row_top[:self.segments*2-1] + '|' + '_'*(self.segments*2) + '|' + '_'*(self.segments*2-1) + '|'
            grid_output.append(row_top)
            
            for i in grid_range:
                row_display = ' '*(self.segments*2-i*2) + '/' + ''.join((' ' + str(self.grid_data[k+x]).ljust(1) + ' /') for x in grid_range)
                k += self.segments
                row_bottom = ' '*(self.segments*2-i*2-1) + '/' + '___/'*self.segments
                
                if j != grid_range[-1]:
                    row_display += ' '*(i*2) + '|'
                    row_bottom += ' '*(i*2+1) + '|'
                if j:
                    row_display = row_display[:self.segments*4+1] + '|' + row_display[self.segments*4+2:]
                    row_bottom = row_bottom[:self.segments*4+1] + '|' + row_bottom[self.segments*4+2:]
                    
                    row_display = '|' + row_display[1:]
                    row_bottom = '|' + row_bottom[1:]
                
                grid_output += [row_display, row_bottom]
                
        return '\n'.join(grid_output)
    
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
        return get_max_dict_keys(self.current_points)
    
    @classmethod
    def from_string(cls, raw_data):
        """Create new Connect3D instance from a string.
        
        Parameters:
            raw_data (str): Passed in from __repr__, 
                contains the grid data and current player.
                Will still work if no player is defined.
                Format: "joined(grid_data).current_player.joined(zfilled(play_data)).joined(zfilled(range_data))"
        """
        split_data = raw_data.split(cls.from_string_separator)
        grid_data = [i if i != ' ' else '' for i in split_data[0]]
        segments = calculate_segments(grid_data)
        longest_number = len(str(pow(segments, 3) - 1))
        
        new_c3d_instance = cls(segments)
        
        new_c3d_instance.grid_data = grid_data
        new_c3d_instance.play_data = None
        new_c3d_instance.range_data = None
        
        #Get current player
        if len(split_data) > 1:
            new_c3d_instance.current_player = split_data[1]
            
        #Get range and play data
        if len(split_data) > 3:
            formatted_play_data = [int(split_data[2][j:j+longest_number]) for j in range(len(split_data[2]))[::longest_number]]
            if all(grid_data[i] for i in formatted_play_data) and formatted_play_data == list(set(formatted_play_data)):
                new_c3d_instance.play_data = formatted_play_data
                
            formatted_range_data = [int(split_data[3][j:j+longest_number]) for j in range(len(split_data[3]))[::longest_number]]
            if sorted(formatted_range_data) == range(pow(segments, 3)):
                new_c3d_instance.range_data = formatted_range_data
        
        new_c3d_instance.update_score()
        
        return new_c3d_instance
        
    @classmethod
    def from_list(cls, grid_data, player=None, play_data=None, range_data=None):
        """Create new Connect3D instance from lists.
        
        Parameters:
            grid_data (list/tuple): 1D list of grid cells, amount must be a cube number.
            
            player (int or None): Current player to continue the game with.
            
            play_data (list or None): List containing the ID of each move currently taken.
                If range_data is None, this will be set to None.
            
            range_data (list or None): List containing the current position of original cell IDs.
                If play_data is None, this will be set to None.
        """
        segments = calculate_segments(grid_data)
        new_c3d_instance = cls(segments)
        
        new_c3d_instance.grid_data = [i if i != ' ' else '' for i in grid_data]
        
        if player is not None:
            new_c3d_instance.current_player = player
        
        if play_data is not None and range_data is not None:
            if not all(grid_data[i] for i in play_data) or not sorted(set(range_data)) == range(pow(segments, 3)):
                play_data = None
                range_data = None
        new_c3d_instance.play_data = play_data
        new_c3d_instance.range_data = range_data
        
        new_c3d_instance.update_score()
        
        return new_c3d_instance
        
    def _old_play(self, p1=False, p2=bot_difficulty_default, shuffle_after=default_shuffle_count, end_when_no_points_left=False):
        """NEEDS TO BE UPDATED
        
        Start or continue a game.
        If using computer players, there is a minimum time delay to avoid it instantly making moves.
        
        Parameters:
            player1 (bool): If player 1 should be played by a computer,
                and if so, what difficulty level.
            
            player2 (bool): If player 2 should be played by a computer,
                and if so, what difficulty level.
        """
        
        players = (p1, p2)
        
        self.current_player = int(not self.current_player)
        move_number = 1
        shuffle_count = 0
        shuffle_after = shuffle_after or self.default_shuffle_count
        min_time_update = 0.65
        
        #Display score and grid
        print
        self.update_score()
        print self.show_score()
        print self
        
        #Game loop
        while True:
            print
            print 'Move {}:'.format(move_number/2)
            move_number += 1
            
            current_time = time.time()
            
            #Switch current player
            self.current_player = int(not self.current_player)
            
            #Check if any points are left to gain
            points_left = True
            if end_when_no_points_left:
                potential_points = {self.player_symbols[j]: Connect3D.from_list([self.player_symbols[j] if i == '' else i for i in self.grid_data]).current_points for j in (0, 1)}
                if any(self.current_points == potential_points[player] for player in (self.player_symbols[j] for j in (0, 1))):
                    points_left = False
                
            #Check if no spaces are left
            if '' not in self.grid_data or not points_left:
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
            
            
            print "Player {}'s turn...".format(self.player_symbols[self.current_player])
            if (p1 == False and not self.current_player) or (p2 == False and self.current_player):
                
                #Player takes a move, function returns True if it updates the grid, otherwise loop again
                while True:
                    player_input = raw_input().replace(',', ' ').replace('.', ' ').split()
                    player_go = self.make_move(self.player_symbols[self.current_player], player_input)
                    if player_go is None:
                        print "Grid cell is not available, try again."
                    else:
                        if self.play_data is not None:
                            self.play_data.append(self.range_data[player_go])
                        break
            else:
                #AI takes a move, will stop the code if it does something wrong
                ai_go = ArtificialIntelligence(self, self.current_player, players[self.current_player]).calculate_next_move()
                if self.make_move(self.player_symbols[self.current_player], ai_go) is None:
                    raise Connect3DError('Something unknown went wrong with the AI')
                else:
                    print "AI moved to point {}.".format(PointConversion(self.segments, ai_go).to_3d())
                    if self.play_data is not None:
                        self.play_data.append(self.range_data[ai_go])
                    
                #Wait a short while
                time.sleep(max(0, min_time_update - (time.time() - current_time)))
    
            
            shuffle_count += 1
            if shuffle_after and shuffle_count >= shuffle_after:
                self.shuffle()
                
            #Display score and grid
            self.update_score()
            print self.show_score()
            print self
            if shuffle_after and shuffle_count >= shuffle_after:
                shuffle_count = 0
                print "Grid was flipped!"

    def play(self, p1_name='Player 1', p2_name='Player 2', 
                   p1=False, p2=bot_difficulty_default, 
                   p1_colour=GREEN, p2_colour=LIGHTBLUE,
                   shuffle_level=1, 
                   end_when_no_points_left=True):
        
        #Need to figure out how to update Connect3D class with result
        self = RunPygame(self).play(p1_name=p1_name, p2_name=p2_name, 
                                    p1_player=p1, p2_player=p2, 
                                    p1_colour=p1_colour, p2_colour=p2_colour,
                                    shuffle_level=shuffle_level, 
                                    end_when_no_points_left=end_when_no_points_left)
                
    def make_move(self, id, *args):
        """Update the grid data with a new move.
        
        Parameters:
            id (str): Character to write into the grid.
            
            args (int, tuple or list): Where in the grid to place the ID.
                Can be input as an integer (grid cell number), 3 integers,
                a tuple or list (3D coordinates)
        
        Returns the ID moved to or None if no move was made.
        
        >>> C3D = Connect3D(2)
        
        >>> C3D.make_move('a', 1)
        1
        >>> C3D.make_move('b', 1)
        >>> C3D.make_move('c', -1)
        >>> C3D.make_move('d', 2, 2, 2)
        7
        >>> C3D.make_move('e', [1, 1, 2])
        4
        >>> C3D.make_move('f', (1, 1, 3))
        
        >>> C3D.grid_data
        ['', 'a', '', '', 'e', '', '', 'd']
        >>> print C3D
             ________
            /   / a /|
           /___/___/ |
          /   /   /  |
         /___/___/   |
        |   |____|___|
        |   / e /|  /
        |  /___/_|_/
        | /   / d|/
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
                    i = PointConversion(self.segments, args[0]).to_int()
            else:
                i = int(args[0])
        else:
            i = PointConversion(self.segments, tuple(args)).to_int()
        
        #Add to grid if cell is empty
        if 0 <= i < self.segments_cubed and self.grid_data[i] not in (0, 1) and i is not None:
            self.grid_data[i] = id
            return i
        else:
            return None
            
            
    def shuffle(self, max_level=2):
        """Mirror the grid in the X, Y, or Z axis.
        
        A max level of 1 is mirror only.
        A max level of 2 includes rotation.
        """
        max_shuffles = max_level * 3
        shuffle_methods = random.sample(range(max_shuffles), random.randint(0, max_shuffles-1))
        if 0 in shuffle_methods:
            self.grid_data = SwapGridData(self.grid_data).fx()
            if self.range_data is not None:
                self.range_data = SwapGridData(self.range_data).fx()
        if 1 in shuffle_methods:
            self.grid_data = SwapGridData(self.grid_data).fy()
            if self.range_data is not None:
                self.range_data = SwapGridData(self.range_data).fy()
        if 2 in shuffle_methods:
            self.grid_data = SwapGridData(self.grid_data).fz()
            if self.range_data is not None:
                self.range_data = SwapGridData(self.range_data).fz()
        if 3 in shuffle_methods:
            reverse = random.randint(0, 1)
            self.grid_data = SwapGridData(self.grid_data).rx(reverse)
            if self.range_data is not None:
                self.range_data = SwapGridData(self.range_data).rx(reverse)
        if 4 in shuffle_methods:
            reverse = random.randint(0, 1)
            self.grid_data = SwapGridData(self.grid_data).ry(reverse)
            if self.range_data is not None:
                self.range_data = SwapGridData(self.range_data).ry(reverse)
        if 5 in shuffle_methods:
            reverse = random.randint(0, 1)
            self.grid_data = SwapGridData(self.grid_data).rz(reverse)
            if self.range_data is not None:
                self.range_data = SwapGridData(self.range_data).rz(reverse)
        self.grid_data.reverse()
        if self.range_data is not None:
            self.range_data.reverse()
            
            
            
    def update_score(self):
        """Recalculate the score.
        
        There are 26 total directions from each point, or 13 lines, calculated in 
        the DirectionCalculation() class. For each of the 13 lines, look both ways
        and count the number of values that match the current player.
        
        This will find any matches from one point, so it's simple to then iterate 
        through every point. A hash of each line is stored to avoid duplicates.
        """
        try:
            self.grid_data_last_updated
        except AttributeError:
            self.grid_data_last_updated = None
        

        if self.grid_data_last_updated != self.grid_data or True:
            
            #Store hash of grid_data in it's current state to avoid unnecessarily running the code again when there's been no changes
            self.grid_data_last_updated = self.grid_data
            
            
            self.current_points = defaultdict(int)
            all_matches = set()
            
            #Loop through each point
            for starting_point in range(len(self.grid_data)):
                
                current_player = self.grid_data[starting_point]
                
                #Fix for the pygame temporary numbers
                current_player_fixed = None
                if type(current_player) == int:
                    current_player_fixed = 9 - current_player
                    
                if current_player != '':
                
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
                        if num_matches == self.segments:
                            list_match = hash(tuple(sorted(list_match)))
                            if list_match not in all_matches:
                                all_matches.add(list_match)
                                self.current_points[current_player] += 1


    def show_score(self, digits=False, marker='/'):
        """Print the current points.
        
        Parameters:
            digits (bool, optional): If the score should be output as a number,
                or as individual marks.
            
            marker (str, optional): How each point should be displayed if 
                digits are not being used.
        
        >>> C3D = Connect3D()
        >>> C3D.update_score()
        >>> C3D.current_points['X'] = 5
        >>> C3D.current_points['O'] = 1
        
        >>> C3D.show_score(False, '/')
        'Player X: /////  Player O: /'
        >>> C3D.show_score(True)
        'Player X: 5  Player O: 1'
        """
        self.update_score()
        multiply_value = 1 if digits else marker
        return 'Player X: {x}  Player O: {o}'.format(x=multiply_value*(self.current_points['X']), o=multiply_value*self.current_points['O'])
    
        
    def reset(self):
        """Empty the grid without creating a new Connect3D object."""
        self.grid_data = ['' for i in range(pow(self.segments, 3))]


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
    Mainly used for inputting the coordinates to make a move.
    
    The cell ID is from 0 to segments^3, and coordinates are from 1 to segments.
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
        segments:
            Size of the grid.
            Type: int
        
        i:
            Cell ID or coordinates.
            Type int/tuple/list
    
    Functions:
        to_3d
        to_int
    """
    def __init__(self, segments, i):
        self.segments = segments
        self.i = i
        
    def to_3d(self):
        """Convert cell ID to a 3D coordinate.
        
        >>> segments = 4
        >>> cell_id = 16
        
        >>> PointConversion(segments, cell_id).to_3d()
        (1, 1, 2)
        """
        cell_id = int(self.i)
        z = cell_id / pow(self.segments, 2) 
        cell_id %= pow(self.segments, 2)
        y = cell_id / self.segments
        x = cell_id % self.segments
        return tuple(cell_id+1 for cell_id in (x, y, z))
    
    def to_int(self):
        """Convert 3D coordinates to the cell ID.
        
        >>> segments = 4
        >>> coordinates = (4,2,3)
        
        >>> PointConversion(segments, coordinates).to_int()
        39
        """
        x, y, z = [int(i) for i in self.i]
        if all(i > 0 for i in (x, y, z)):
            return (x-1)*pow(self.segments, 0) + (y-1)*pow(self.segments, 1) + (z-1)*pow(self.segments, 2)
        return None


class SwapGridData(object):
    """Use the size of the grid to calculate how flip it on the X, Y, or Z axis.
    The flips keep the grid intact but change the perspective of the game.
    
    Parameters:
        grid_data (list/tuple): 1D list of grid cells, amount must be a cube number.
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
        """Flip on the X axis.
        
        >>> SwapGridData(range(8)).fx()
        [1, 0, 3, 2, 5, 4, 7, 6]
        >>> print Connect3D.from_list(SwapGridData(range(8)).fx())
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
        return join_list(x[::-1] for x in split_list(self.grid_data, self.n))
        
    def fy(self):
        """Flip on the Y axis.
        
        >>> SwapGridData(range(8)).fy()
        [2, 3, 0, 1, 6, 7, 4, 5]
        >>> print Connect3D.from_list(SwapGridData(range(8)).fy())
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
        return join_list(join_list(split_list(x, self.n)[::-1]) for x in self.group_split)
        
    def fz(self):
        """Flip on the Z axis.
        
        >>> SwapGridData(range(8)).fz()
        [4, 5, 6, 7, 0, 1, 2, 3]
        >>> print Connect3D.from_list(SwapGridData(range(8)).fz())
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
        return join_list(split_list(self.grid_data, pow(self.n, 2))[::-1])
    
    def rx(self, reverse=None):
        """Rotate on the X axis.
        
        >>> SwapGridData(range(8)).rx(False)
        [5, 1, 7, 3, 4, 0, 6, 2]
        >>> print Connect3D.from_list(SwapGridData(range(8)).rx(False))
             ________
            / 5 / 1 /|
           /___/___/ |
          / 7 / 3 /  |
         /___/___/   |
        |   |____|___|
        |   / 4 /|0 /
        |  /___/_|_/
        | / 6 / 2|/
        |/___/___|
        
        >>> SwapGridData(range(8)).rx(True)
        [1, 5, 3, 7, 0, 4, 2, 6]
        >>> print Connect3D.from_list(SwapGridData(range(8)).rx(True))
             ________
            / 1 / 5 /|
           /___/___/ |
          / 3 / 7 /  |
         /___/___/   |
        |   |____|___|
        |   / 0 /|4 /
        |  /___/_|_/
        | / 2 / 6|/
        |/___/___|
        """
        n_sq = pow(self.n, 2)
        n_start = pow(self.n, 3) - n_sq
        
        if reverse is None:
            reverse = random.randint(0, 1)
        
        if reverse:
            return [self.grid_data[n_start + i + j * self.n - k * n_sq] for i in self.range_reverse for j in self.range for k in self.range_reverse]
        else:
            return [self.grid_data[n_start + i + j * self.n - k * n_sq] for i in self.range for j in self.range for k in self.range]
            
    def ry(self, reverse=None):
        """Rotate on the Y axis.
        
        >>> SwapGridData(range(8)).ry(False)
        [4, 5, 0, 1, 6, 7, 2, 3]
        >>> print Connect3D.from_list(SwapGridData(range(8)).ry(False))
             ________
            / 4 / 5 /|
           /___/___/ |
          / 0 / 1 /  |
         /___/___/   |
        |   |____|___|
        |   / 6 /|7 /
        |  /___/_|_/
        | / 2 / 3|/
        |/___/___|
        
        >>> SwapGridData(range(8)).ry(True)
        [2, 3, 6, 7, 0, 1, 4, 5]
        >>> print Connect3D.from_list(SwapGridData(range(8)).ry(True))
             ________
            / 2 / 3 /|
           /___/___/ |
          / 6 / 7 /  |
         /___/___/   |
        |   |____|___|
        |   / 0 /|1 /
        |  /___/_|_/
        | / 4 / 5|/
        |/___/___|
        """
        if reverse is None:
            reverse = random.randint(0, 1)
        if reverse:
            return join_list(j[offset:offset + self.n] for offset in [(self.n - i - 1) * self.n for i in self.range] for j in self.group_split)
        else:
            gs_reverse = self.group_split[::-1]
            return join_list(j[offset:offset + self.n] for offset in [i * self.n for i in self.range] for j in gs_reverse)
            
    def rz(self, reverse=None):
        """Rotate on the Z axis.
        
        >>> SwapGridData(range(8)).rz(False)
        [2, 0, 3, 1, 6, 4, 7, 5]
        >>> print Connect3D.from_list(SwapGridData(range(8)).rz(False))
             ________
            / 2 / 0 /|
           /___/___/ |
          / 3 / 1 /  |
         /___/___/   |
        |   |____|___|
        |   / 6 /|4 /
        |  /___/_|_/
        | / 7 / 5|/
        |/___/___|
        
        >>> SwapGridData(range(8)).rz(True)
        [1, 3, 0, 2, 5, 7, 4, 6]
        >>> print Connect3D.from_list(SwapGridData(range(8)).rz(True))
             ________
            / 1 / 3 /|
           /___/___/ |
          / 0 / 2 /  |
         /___/___/   |
        |   |____|___|
        |   / 5 /|7 /
        |  /___/_|_/
        | / 4 / 6|/
        |/___/___|
        """
        if reverse is None:
            reverse = random.randint(0, 1)
            
        if reverse:
            return [x[j][i] for x in [split_list(x, self.n) for x in self.group_split] for i in self.range_reverse for j in self.range]
        else:
            return [x[j][i] for x in [split_list(x, self.n)[::-1] for x in self.group_split] for i in self.range for j in self.range]
    
    def reverse(self):
        """Reverse the grid.
        
        >>> SwapGridData(range(8)).reverse()
        [7, 6, 5, 4, 3, 2, 1, 0]
        >>> print Connect3D.from_list(SwapGridData(range(8)).reverse())
             ________
            / 7 / 6 /|
           /___/___/ |
          / 5 / 4 /  |
         /___/___/   |
        |   |____|___|
        |   / 3 /|2 /
        |  /___/_|_/
        | / 1 / 0|/
        |/___/___|
        """
        return self.grid_data[::-1]


def calculate_segments(grid_data):
    """Cube root the length of grid_data to find the grid size."""
    return int(round(pow(len(grid_data), 1.0/3.0), 0))


def split_list(x, n):
    """Split a list by n characters."""
    n = int(n)
    return [x[i:i+n] for i in range(0, len(x), n)]
    
    
def join_list(x):
    """Convert nested lists into one single list."""
    return [j for i in x for j in i]
        
        
def get_max_dict_keys(x):
    """Return a list of every key containing the max value.
    
    Parameters:
        x (dict): Dictionary to sort and get highest value.
            It must be a dictionary of integers to work properly.
    """
    if x:
        sorted_dict = sorted(x.iteritems(), key=operator.itemgetter(1), reverse=True)
        if sorted_dict[0][1]:
            return sorted([k for k, v in x.iteritems() if v == sorted_dict[0][1]])
    return []
  
  
class ArtificialIntelligence(object):
    """AI coded to play Connect3D."""
    
    def __init__(self, C3DObject, player_num, difficulty=Connect3D.bot_difficulty_default):
        """Set up the AI for a single move using the current state of Connect3D.
        
        Parameters:
            C3DObject (object): Connect3D object, needed to get the current 
                state of the game as opposed to passing in lots of values.
            
            player_num (int): Which player the AI is.
            
            difficulty (string/int): Difficulty level to use for the AI.
        """
        self.C3DObject = C3DObject
        self.player_num = player_num
        self.player = player_num #Connect3D.player_symbols[1]
        self.enemy = int(not self.player_num)# Connect3D.player_symbols[int(not self.player_num)]
        self.gd_len = self.C3DObject.segments_cubed
        self.difficulty = difficulty
        self.grid_data = [i if i in (self.player, self.enemy) else '' for i in C3DObject.grid_data]
        
        self.checks = 0
    
    def max_cell_points(self):
        """Get maximum number of points that can be gained from each empty cell,
        that is not blocked by an enemy value.
        """
        max_points = defaultdict(int)
        filled_grid_data = [i if i != '' else self.player for i in self.grid_data]
        for cell_id in range(self.gd_len):
            self.checks += 1            
            if filled_grid_data[cell_id] == self.player and self.grid_data[cell_id] == '':
                max_points[cell_id] = self.check_grid(filled_grid_data, cell_id, self.player)
                
        return get_max_dict_keys(max_points)
    
    def check_for_n_minus_one(self, grid_data=None):
        """Find all places where anyone has n-1 points in a row, by substituting
        in a point for each player in every cell.
        
        Parameters:
            grid_data (list or None, optional): Pass in a custom grid_data, 
                leave as None to use the Connect3D one.
        """
        
        if grid_data is None:
            grid_data = list(self.grid_data)
        
        matches = defaultdict(list)
        for cell_id in range(len(grid_data)):
            if grid_data[cell_id] == '':
                for current_player in (self.player, self.enemy):
                    if self.check_grid(grid_data, cell_id, current_player):
                        matches[current_player].append(cell_id)
        return matches
    
    def look_ahead(self):
        """Look two moves ahead to detect if someone could get a point.
        Uses the check_for_n_minus_one function from within a loop.
        
        Will return 1 as the second parameter if it has looked up more than a single move.
        """
        #Try initial check
        match = self.check_for_n_minus_one()
        if match:
            return (match, 0)
            
        #For every grid cell, substitute a player into it, then do the check again
        grid_data = list(self.grid_data)
        matches = defaultdict(list)
        for i in range(self.gd_len):
            if self.C3DObject.grid_data[i] == '':
                old_value = grid_data[i]
                for current_player in (self.player, self.enemy):
                    grid_data[i] = current_player
                    match = self.check_for_n_minus_one(grid_data)
                    if match:
                        for k, v in match.iteritems():
                            matches[k] += v
                            
                grid_data[i] = old_value
                
        if matches:
            return (matches, 1)
            
        return (defaultdict(list), 0)
    
    def check_grid(self, grid_data, cell_id, player):
        """Duplicate of the Connect3D.update_score method, but set up to check individual cells.
        
        Parameters:
            grid_data (list/tuple): 1D list of grid cells, amount must be a cube number.
            
            cell_id (int): The cell ID, or grid_data index to update.
            
            player (int): Integer representation of the player, can be 0 or 1.
        """
        max_points = 0
        for i in DirectionCalculation().opposite_direction:
            
            #Get a list of directions and calculate movement amount
            possible_directions = [list(i)]
            possible_directions += [[j.replace(i, '') for i in possible_directions[0] for j in DirectionCalculation().direction_group.values() if i in j]]
            direction_movement = sum(self.C3DObject.direction_maths[j] for j in possible_directions[0])
            
            #Build list of invalid directions
            invalid_directions = [[self.C3DObject.direction_edges[j] for j in possible_directions[k]] for k in (0, 1)]
            invalid_directions = [join_list(j) for j in invalid_directions]
            
            num_matches = 1
            
            #Use two loops for the opposite directions
            for j in (0, 1):
                
                current_point = cell_id
                
                while current_point not in invalid_directions[j] and 0 <= current_point < len(grid_data):
                
                    self.checks += 1
                    current_point += direction_movement * int('-'[:j] + '1')
                    if grid_data[current_point] == player:
                        num_matches += 1
                    else:
                        break

            #Add a point if enough matches
            if num_matches == self.C3DObject.segments:
                max_points += 1
                     
        return max_points
    
    def calculate_next_move(self):
        """Groups together the AI methods in order of importance.
        Will throw an error if grid_data is full, since the game should have ended by then anyway.
        
        The far_away part determins which order to do things in.
        
            It's set up so that for n-1 in a row, the most urgent thing is to stop the 
            opposing player before gaining any points. However, for n-2 in a row, it's 
            more useful to gain points if possible.
            
            By setting order_of_importance to 0, it'll always try block the player 
            first, and by setting to 1, it'll always try score points regardless of 
            if the other player will get one too.
        """
        
        chance_of_changing_tactic, chance_of_not_noticing, chance_of_not_noticing_divide = get_bot_difficulty(self.difficulty)
        
        grid_data_joined_len = len(''.join(map(str, self.C3DObject.grid_data)))
        
        next_moves = []
        self.C3DObject.ai_message = []
        ai_message = self.C3DObject.ai_message.append
        self.checks = 0
        state = None
        
        if grid_data_joined_len > (self.C3DObject.segments - 2) * 2 - 1:
            
            point_based_move, far_away = self.look_ahead()
            
            #Format debug message
            possible_moves_type = ['Block', 'Gain']
            if not self.player:
                possible_moves_type = possible_moves_type[::-1]
            for k, v in point_based_move.iteritems():
                if k == self.player:
                    message = 'Gain Moves: {}'
                elif k == self.enemy:
                    message = 'Block Moves: {}'
                self.C3DObject.ai_message.append(message.format(v))
                    
            
            #self.C3DObject.ai_message.append('Possible Moves: {}'.format(possible_moves_message))
            self.C3DObject.ai_message.append('Urgent: {}'.format(not far_away))
            
            #Reduce chance of not noticing n-1 in a row, since n-2 in a row isn't too important
            if not far_away:
                chance_of_not_noticing /= chance_of_not_noticing_divide
                chance_of_not_noticing = pow(chance_of_not_noticing, pow(grid_data_joined_len / float(len(self.C3DObject.grid_data)), 0.4))
            
            
            #Set which order to do things in
            order_of_importance = int('-'[:int(far_away)] + '1')
            
            ai_new_tactic = random.uniform(0, 100) < chance_of_changing_tactic
            if ai_new_tactic:
                ai_message('AI changed tacic.')
                order_of_importance = random.choice((-1, 1))
                
            move1_player = [self.enemy, self.player][::order_of_importance]
            move1_text = ['Blocking opposing player', 'Gaining points'][::order_of_importance]
            
            
            #Predict if other player is trying to trick the AI
            #(or try trick the player if the setup is right)
            #Quite an advanced tactic so chance of not noticing is increased
            if min(random.uniform(0, 100), random.uniform(0, 100)) > chance_of_not_noticing:
            
                matching_ids = [defaultdict(int) for i in range(3)]
                for k, v in point_based_move.iteritems():
                    for i in v:
                        matching_ids[k][i] += 1
                        matching_ids[2][i] += 1
                
                if matching_ids[2]:
                    
                    if matching_ids[move1_player[0]]:
                        highest_occurance = max(matching_ids[move1_player[0]].iteritems(), key=operator.itemgetter(1))[1]
                        if highest_occurance > 1:
                            next_moves = [k for k, v in matching_ids[move1_player[0]].iteritems() if v == highest_occurance]
                            state = 'Forward thinking ({})'.format(move1_text[0])
                        
                    if not next_moves and matching_ids[move1_player[1]]:
                        highest_occurance = max(matching_ids[move1_player[1]].iteritems(), key=operator.itemgetter(1))[1]
                        if highest_occurance > 1:
                            next_moves = [k for k, v in matching_ids[move1_player[1]].iteritems() if v == highest_occurance]
                            state = 'Forward thinking ({})'.format(move1_text[1])
                    
                    if not next_moves:
                        highest_occurance = max(matching_ids[2].iteritems(), key=operator.itemgetter(1))[1]
                        if highest_occurance > 1:
                            next_moves = [k for k, v in matching_ids[2].iteritems() if v == highest_occurance]
                            state = 'Forward thinking'
            
            
            #Make a move based on other points
            ai_noticed = random.uniform(0, 100) > chance_of_not_noticing
            if point_based_move and ai_noticed and (not next_moves or not far_away):
                if point_based_move[move1_player[0]]:
                    next_moves = point_based_move[move1_player[0]]
                    state = move1_text[0]
                    
                elif point_based_move[move1_player[1]]:
                    next_moves = point_based_move[move1_player[1]]
                    state = move1_text[1]
            
            #Make a random move determined by number of possible points
            elif not state:
                if not ai_noticed:
                    ai_message("AI didn't notice something.")
                state = False
                
        #Make a semi random placement
        if not state:
            if not chance_of_not_noticing and random.uniform(0, 100) > chance_of_not_noticing:
                next_moves = self.max_cell_points()
                state = 'Predictive placement'
            else:
                state = 'Random placement'
            
            if state is None:
                state = 'Starting'
            
        #Make a totally random move
        if not next_moves:
            next_moves = [i for i in range(self.gd_len) if self.grid_data[i] == '']
            if state is None:
                state = 'Struggling'
            
        
        ai_message('AI Objective: {}.'.format(state))
        n = random.choice(next_moves)
        
        if len(next_moves) != len(self.grid_data) - len(''.join(map(str, self.grid_data))):
            ai_message('Potential Moves: {}'.format(next_moves))
        
        next_move = random.choice(next_moves)
        
        ai_message('Chosen Move: {}'.format(next_move))
        ai_message('Calculations: {}'.format(self.checks + 1))
        
        return next_move

        
def get_bot_difficulty(level, _default=Connect3D.bot_difficulty_default, _debug=False):
    """Preset parameters for the bot difficulty levels.
    
    Parameters:
        level (str/int): Difficulty level to get the data for.
        
        _default (str/int): If level is invalid, use this as the default value.
    
    There are 3 variables to control the chance of doing something differently:
        Changing Tactic - Normally the computer will give priority to blocking an
            enemy row of n-1 before completing it's own, and adding to a row of
            n-2 before blocking the enemy. This is the percent chance to override
            this behavior.
        
        Not Noticing - The chance the computer will miss a row that is almost complete.
            Without this, the computer will be able to block absolutely everything 
            unless it is tricked.
            Leave this quite high for the n-2 rows, since it can be frustrating to
            have every row blocked before you've even half finished it.
        
        Not Noticing Divide - Not noticing rows of n-2 keeps the game flowing, 
            not noticing rows of n-1 makes it too easy to win. This will reduce the
            'Not Noticing' chance for rows of n-1 so the computer doesn't look
            like it's totally blind.
        
        In addition to the 'Not Noticing Divide', as the grid is more empty and is
        easy to see, the chance of not noticing something is reduced.    
    """
    difficulty_level = {}
    difficulty_level[0] = 'beginner'
    difficulty_level[1] = 'easy'
    difficulty_level[2] = 'medium'
    difficulty_level[3] = 'hard'
    difficulty_level[4] = 'extreme'
    
    if level is False and _debug:
        return 0
        
    try:
        level = difficulty_level[level]
    except KeyError:
        level = str(level).lower()
        
    if level == difficulty_level[0]:
        if _debug:
            return 1
        return (75, 95, 1)
    elif level == difficulty_level[1]:
        if _debug:
            return 2
        return (50, 75, 2)
    elif level == difficulty_level[2]:
        if _debug:
            return 3
        return (40, 40, 4)
    elif level == difficulty_level[3]:
        if _debug:
            return 4
        return (20, 20, 4)
    elif level == difficulty_level[4]:
        if _debug:
            return 5
        return (0, 0, 1)
    
    return get_bot_difficulty(_default, _debug)


class CoordinateConvert(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.centre = (self.width / 2, self.height / 2)

    def to_pygame(self, x, y):
        x = x - self.centre[0]
        y = self.centre[1] - y
        return (x, y)

    def to_canvas(self, x, y):
        x = x + self.centre[0]
        y = self.centre[1] - y
        return (x, y)

        
class GridDrawData(object):
    """Hold the relevant data for the grid, to allow it to be shown."""
    
    def __init__(self, length, segments, angle, PADDING_TEXT=5, offset=(0, 0)):
        self.length = length
        self.segments = segments
        self.angle = angle
        self.PADDING_TEXT = PADDING_TEXT
        self.offset = list(offset)
        self._calculate()

    def game_to_block_index(self, gx, gy):
        """Return index of block at the game coordinates gx, gy, or None if
        there is no block at those coordinates."""
        gx -= self.offset[0]
        gy -= self.offset[1] - self.centre
        z = int(gy // self.chunk_height)
        gy -= z * self.chunk_height
        
        dx = gx / self.size_x_sm
        dy = gy / self.size_y_sm
        x = int((dy - dx) // 2)
        y = int((dy + dx) // 2)
        
        n = self.segments
        if 0 <= x < n and 0 <= y < n and 0 <= z < n:
            return n ** 3 - 1 - (x + n * (y + n * z))
        else:
            return None
            
    def block_index(self, x, y, z):
        """Return the block index corresponding to the block at x, y, z, or
        None if there is no block at those coordinates.

        """
        n = self.segments
        if 0 <= x < n and 0 <= y < n and 0 <= z < n:
            return n ** 3 - 1 - (x + n * (y + n * z))
        else:
            return None
            
    def block_to_game(self, x, y, z):
        """Return the game coordinates corresponding to block x, y, z."""
        gx = (x - y) * self.size_x_sm
        gy = (x + y) * self.size_y_sm + z * self.chunk_height - self.centre
        return gx, gy

    def _calculate(self):
        """Perform the main calculations on the values in __init__.
        This allows updating any of the values, such as the isometric
        angle, without creating a new class."""
        
        self.size_x = self.length * math.cos(math.radians(self.angle))
        self.size_y = self.length * math.sin(math.radians(self.angle))
        self.x_offset = self.size_x / self.segments
        self.y_offset = self.size_y / self.segments
        self.chunk_height = self.size_y * 2 + self.PADDING_TEXT
        
        self.centre = (self.chunk_height / 2) * self.segments - self.PADDING_TEXT / 2
        self.size_x_sm = self.size_x / self.segments
        self.size_y_sm = self.size_y / self.segments
        
        self.length_small = self.length / self.segments
        
        self.relative_coordinates = []
        position = (0, self.centre)
        for j in range(self.segments):
            checkpoint = position
            for i in range(self.segments):
                self.relative_coordinates.append(position)
                position = (position[0] + self.x_offset,
                            position[1] - self.y_offset)
            position = (checkpoint[0] - self.x_offset,
                        checkpoint[1] - self.y_offset)



        #Absolute coordinates for pygame
        chunk_coordinates = [(self.offset[0], self.offset[1] - i * self.chunk_height) for i in range(self.segments)]

        self.line_coordinates = [((self.offset[0] + self.size_x, self.offset[1] + self.centre - self.size_y),
                                  (self.offset[0] + self.size_x, self.offset[1] + self.size_y - self.centre)),
                                 ((self.offset[0] - self.size_x, self.offset[1] + self.centre - self.size_y),
                                  (self.offset[0] - self.size_x, self.offset[1] + self.size_y - self.centre)),
                                 ((self.offset[0], self.offset[1] + self.centre - self.size_y * 2),
                                  (self.offset[0], self.offset[1] - self.centre))]

        for i in range(self.segments):

            chunk_height = -i * self.chunk_height

            self.line_coordinates += [((self.offset[0] + self.size_x, self.offset[1] + self.centre + chunk_height - self.size_y),
                                       (self.offset[0], self.offset[1] + self.centre + chunk_height - self.size_y * 2)),
                                      ((self.offset[0] - self.size_x, self.offset[1] + self.centre + chunk_height - self.size_y),
                                       (self.offset[0], self.offset[1] + self.centre + chunk_height - self.size_y * 2))]

            for coordinate in self.relative_coordinates:
                
                start = (self.offset[0] + coordinate[0], self.offset[1] + chunk_height + coordinate[1])
                self.line_coordinates += [(start,
                                           (start[0] + self.size_x_sm, start[1] - self.size_y_sm)),
                                          (start,
                                           (start[0] - self.size_x_sm, start[1] - self.size_y_sm))]


class RunPygame(object):
    
    overlay_marker = '/'
    EMPTY = YELLOW
    FPS_IDLE = 15
    FPS_MAIN = 30
    FPS_SMOOTH = 120
    PADDING_TEXT = (5, 10)
    OVERLAY_WIDTH = 500
    PADDING_OPTIONS = 2
    PLAYER_RANGE = (0, 1)
    
    def __init__(self, C3DObject, screen_width=640, screen_height=860, default_length=200, default_angle=24):
        self.C3DObject = C3DObject
        self.width = screen_width
        self.height = screen_height
        self.length = default_length
        self.angle = default_angle
        self.player = int(not self.C3DObject.current_player)
        
        
    def _next_player(self):
        self.player = int(not self.player)
    
    def _previous_player(self):
        self._next_player()
    
    def _set_fps(self, fps):
        try:
            if self._fps is None or self._fps < fps:
                self._fps = fps
        except AttributeError:
            self._fps = None
            self._set_fps(fps)
    
    def _update_window_size(self):
        self.convert = CoordinateConvert(self.width, self.height)
        self.to_pygame = self.convert.to_pygame
        self.to_canvas = self.convert.to_canvas
        
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.backdrop = pygame.Surface((self.width, self.height))
        self.backdrop.set_alpha(196)
        self.backdrop.fill(WHITE)
    
    def play(self, p1_name, p2_name, p1_player, p2_player, p1_colour, p2_colour, shuffle_level, end_when_no_points_left):
    
        #Setup pygame
        pygame.init()
        self._update_window_size()
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('Connect 3D')
        background_colour = BACKGROUND
        self.player_names = (p1_name, p2_name)
        self.player_colours = [p1_colour, p2_colour]
        offset = (0, -25)
        
        #Import the font
        self.font_file = 'Miss Monkey.ttf'
        try:
            pygame.font.Font(self.font_file, 0)
        except IOError:
            raise IOError('unable to load font - download from http://www.dafont.com/miss-monkey.font')
        self.font_lg = pygame.font.Font(self.font_file, 36)
        self.font_lg_size = self.font_lg.render('', 1, BLACK).get_rect()[3]
        self.font_md = pygame.font.Font(self.font_file, 24)
        self.font_md_size = self.font_md.render('', 1, BLACK).get_rect()[3]
        self.font_sm = pygame.font.Font(self.font_file, 18)
        self.font_sm_size = self.font_sm.render('', 1, BLACK).get_rect()[3]
            
        self.draw_data = GridDrawData(self.length,
                                 self.C3DObject.segments,
                                 self.angle,
                                 PADDING_TEXT=self.angle / self.C3DObject.segments,
                                 offset=offset)
        
        game_data = {'players': (p1_player, p2_player),
                     'overlay': 'options',
                     'move_number': 0,
                     'shuffle': [shuffle_level, 3],
                     'debug': False,
                     'offset': offset}
        misc_data = {'pending_move_start': 0,
                     'shuffle_count': 0,
                     'temp_fps': self.FPS_MAIN,
                     'resize_start_x': 0,
                     'resize_start_y': 0,
                     'resize_end_x': 0,
                     'resize_end_y': 0}
        tick_data = {'old': 0,
                     'new': 0,
                     'update': 4, #How many ticks between each held key command
                     'total': 0}
        mouse_hover = {'player': None,
                       'shuffle': None,
                       'debug': None,
                       'segments': None,
                       'end_early': end_when_no_points_left,
                       'quit': False,
                       'continue': False,
                       'new': False,
                       'instructions': False}
        held_keys = {'angle': 0,
                     'size': 0,
                     'x': 0,
                     'y': 0}
                     
        mouse_data = pygame.mouse.get_pos()
                     
        #How long to wait before accepting a move
        moving_wait = 0.6
        
        #For controlling how the angle and length of grid change
        angle_increment = 0.25
        angle_max = 40
        length_exponential = 1.1
        length_increment = 0.5
        length_multiplier = 0.01
        offset_increment = 0.025 #Speed of movement
        offset_exponential = 0.75 #Determins ratio of speed to length
        offset_smooth = 4 #How fast it hits full speed
        offset_falloff = 2 #How fast it stops
        time_update = 0.01
        
        self._set_fps(self.FPS_MAIN)
        segments = self.C3DObject.segments
        game_flags = {'reset': True}
        while True:
            self.clock.tick(self._fps or self.FPS_IDLE)
            tick_data['new'] = pygame.time.get_ticks()
            frame_time = time.time()
            tick_data['total'] += 1
            
            #Reinitialise the game
            if game_flags['reset']:
                game_data['move_number'] = 0
                game_data['shuffle'][0] = shuffle_level
                game_data['players'] = (p1_player, p2_player)
                self.C3DObject.segments = segments
                self.C3DObject = Connect3D(self.C3DObject.segments)
                
                block_data = {'id': None,
                              'taken': False}
                
                game_flags = {'pending_move': None,
                              'clicked': False,
                              'mouse_used': True,
                              'quit': False,
                              'recalculate': False,
                              'reset': False,
                              'hover': False,
                              'flipped': False,
                              'disable_background_clicks': False,
                              'winner': None,
                              'points_left': True}
            
            #Reset game loop
            game_flags['recalculate'] = False
            game_flags['mouse_used'] = False
            game_flags['clicked'] = False
            game_flags['disable_background_clicks'] = False
            self._fps = None
                
            if game_flags['quit']:
                return self.C3DObject
                
            #Check if no spaces are left
            if all(i in (0, 1) for i in self.C3DObject.grid_data) or not game_flags['points_left']:
                if not game_flags['winner']:
                    game_flags['winner'] = self.C3DObject._get_winning_player()
                    player_difficulties = [get_bot_difficulty(i, _debug=True) for i in game_data['players']]
                    game_data['overlay'] = 'gameend'
                    
            #Delete the hover data
            if game_flags['hover'] is not None:
                if self.C3DObject.grid_data[game_flags['hover']] == self.overlay_marker:
                    self.C3DObject.grid_data[game_flags['hover']] = ''
                game_flags['hover'] = None
            
            if game_data['overlay']:
                game_flags['disable_background_clicks'] = True
            
            #Delay each go
            if game_flags['pending_move']:
                game_flags['disable_background_clicks'] = True
                
                if misc_data['pending_move_start'] < frame_time:
                
                    attempted_move = self.C3DObject.make_move(*game_flags['pending_move'])
                    
                    if attempted_move is not None:
                        game_data['move_number'] += 1
                        self.C3DObject.update_score()
                        misc_data['shuffle_count'] += 1
                        
                        #Check if any points are left
                        if end_when_no_points_left:
                            for player in self.PLAYER_RANGE:
                                potential_points = {player: Connect3D.from_list([player if i not in self.PLAYER_RANGE else i 
                                                                                 for i in self.C3DObject.grid_data]).current_points
                                                    for player in self.PLAYER_RANGE}
                                game_flags['points_left'] = any(v != self.C3DObject.current_points for v in potential_points.values())
                        
                        #Shuffle grid
                        if misc_data['shuffle_count'] >= game_data['shuffle'][1] and game_data['shuffle'][0]:
                            misc_data['shuffle_count'] = 0
                            self.C3DObject.shuffle(game_data['shuffle'][0])
                            game_flags['flipped'] = True
                        else:
                            game_flags['flipped'] = False
                            
                    else:
                        self._next_player()
                        print "Invalid move: {}".format(game_flags['pending_move'][1])
                        
                    game_flags['pending_move'] = None
                    game_flags['mouse_used'] = True
                    
                    #Skip to end of game if last move
                    if len(''.join(map(str, self.C3DObject.grid_data))) == self.C3DObject.segments_cubed:
                        continue
                    
                else:
                    try:
                        self.C3DObject.grid_data[game_flags['pending_move'][1]] = 9 - game_flags['pending_move'][0]
                    except TypeError:
                        print game_flags['pending_move'], ai_turn
                        raise TypeError('something went wrong, trying to find the cause of this')
                
            #Run the AI
            ai_turn = None
            if game_data['players'][self.player] is not False:
                if not game_flags['disable_background_clicks'] and game_flags['winner'] is None:
                    ai_turn = ArtificialIntelligence(self.C3DObject, self.player, difficulty=game_data['players'][self.player]).calculate_next_move()
                
            
            #Event loop
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    return
                
                if event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.dict['size']
                    self._update_window_size()

                #Get single key presses
                if event.type == pygame.KEYDOWN:
                    game_flags['recalculate'] = True

                    #Set what the escape key does
                    if event.key == pygame.K_ESCAPE:
                        if game_data['overlay'] is None:
                            game_data['overlay'] = 'options'
                        elif game_data['overlay'] == 'gameend':
                            pass
                        elif game_data['overlay'] == 'instructions':
                            game_data['overlay'] = 'options'
                        else:
                            game_data['overlay'] = None
                    
                    #Manually flip the grid
                    if event.key == pygame.K_f:
                        if game_data['debug']:
                            if game_flags['pending_move']:
                                old_move_index = self.C3DObject.range_data[game_flags['pending_move'][1]]
                            misc_data['shuffle_count'] = 0
                            self.C3DObject.shuffle()
                            game_flags['flipped'] = True
                            if game_flags['pending_move']:
                                game_flags['pending_move'][1] = self.C3DObject.range_data.index(old_move_index)
                    
                    #Resize the grid
                    if event.key == pygame.K_w:
                        held_keys['angle'] = 1

                    if event.key == pygame.K_s:
                        held_keys['angle'] = -1

                    if event.key == pygame.K_d:
                        held_keys['size'] = 1

                    if event.key == pygame.K_a:
                        held_keys['size'] = -1
                    
                    #Reset the grid
                    if event.key == pygame.K_c:
                        self.draw_data.offset = list(offset)
                        misc_data['resize_end_x'] = misc_data['resize_end_y'] = 0
                        self.draw_data.angle = self.angle
                        self.draw_data.length = self.length
                    
                    #Move the grid
                    if event.key == pygame.K_UP:
                        held_keys['y'] = 1
                        misc_data['resize_start_y'] = frame_time
                    
                    if event.key == pygame.K_DOWN:
                        held_keys['y'] = -1
                        misc_data['resize_start_y'] = frame_time
                        
                    if event.key == pygame.K_LEFT:
                        held_keys['x'] = -1
                        misc_data['resize_start_x'] = frame_time

                    if event.key == pygame.K_RIGHT:
                        held_keys['x'] = 1
                        misc_data['resize_start_x'] = frame_time
                
                #Fix for opposite buttons overriding each other
                if event.type == pygame.KEYUP:
                    key = pygame.key.get_pressed()
                    if event.key == pygame.K_w and key[pygame.K_s]:
                        held_keys['angle'] = -1

                    if event.key == pygame.K_s and key[pygame.K_w]:
                        held_keys['angle'] = 1

                    if event.key == pygame.K_d and key[pygame.K_a]:
                        held_keys['size'] = -1

                    if event.key == pygame.K_a and key[pygame.K_d]:
                        held_keys['size'] = 1
                        
                    if event.key == pygame.K_UP and key[pygame.K_DOWN]:
                        held_keys['y'] = -1
                        misc_data['resize_start_y'] = frame_time
                    
                    if event.key == pygame.K_DOWN and key[pygame.K_UP]:
                        held_keys['y'] = 1
                        misc_data['resize_start_y'] = frame_time
                        
                    if event.key == pygame.K_LEFT and key[pygame.K_RIGHT]:
                        held_keys['x'] = 1
                        misc_data['resize_start_x'] = frame_time

                    if event.key == pygame.K_RIGHT and key[pygame.K_LEFT]:
                        held_keys['x'] = -1
                        misc_data['resize_start_x'] = frame_time
                        
                #Get mouse clicks
                if event.type == pygame.MOUSEBUTTONDOWN:
                    game_flags['clicked'] = event.button
                    game_flags['mouse_used'] = True
                
                if event.type == pygame.MOUSEMOTION:
                    game_flags['mouse_used'] = True
            
            
            #Get held down key presses, but only update if enough ticks have passed
            key = pygame.key.get_pressed()
            update_yet = False
            if tick_data['new'] - tick_data['old'] > tick_data['update']:
                update_yet = True
                tick_data['old'] = pygame.time.get_ticks()
            
            changed_something = False
            if held_keys['angle']:
                
                if not (key[pygame.K_w] or key[pygame.K_s]) or (key[pygame.K_w] and key[pygame.K_s]):
                    held_keys['angle'] = 0
                
                elif update_yet:
                    self.draw_data.angle += angle_increment * held_keys['angle']
                    changed_something = True
            
            if held_keys['size']:
                if not (key[pygame.K_a] or key[pygame.K_d]) or (key[pygame.K_a] and key[pygame.K_d]):
                    held_keys['size'] = 0
                    
                elif update_yet:
                    length_exp = (max(length_increment,
                                     (pow(self.draw_data.length, length_exponential)
                                      - 1 / length_increment))
                                  * length_multiplier)
                    self.draw_data.length += length_exp * held_keys['size']
                    changed_something = True

                    
            #Move the grid on the screen
            if held_keys['x']:
                time_difference = min(1.0 / offset_smooth, frame_time - misc_data['resize_start_x']) * offset_smooth
                if not (key[pygame.K_LEFT] or key[pygame.K_RIGHT]) or (key[pygame.K_LEFT] and key[pygame.K_RIGHT]):
                    misc_data['resize_end_x'] = (frame_time + max(1.0 / offset_smooth, time_difference) - 1) * held_keys['x']
                    held_keys['x'] = 0
                
                elif update_yet:
                    self.draw_data.offset[0] += offset_increment * held_keys['x'] * pow(self.draw_data.length, offset_exponential) * time_difference
                    changed_something = True
            
            if held_keys['y']:
                time_difference = min(1.0 / offset_smooth, frame_time - misc_data['resize_start_y']) * offset_smooth
                if not (key[pygame.K_UP] or key[pygame.K_DOWN]) or (key[pygame.K_LEFT] and key[pygame.K_RIGHT]):
                    misc_data['resize_end_y'] = (frame_time + max(1.0 / offset_smooth, time_difference) - 1) * held_keys['y']
                    held_keys['y'] = 0
                
                elif update_yet:
                    self.draw_data.offset[1] += offset_increment * held_keys['y'] * pow(self.draw_data.length, offset_exponential) * time_difference
                    changed_something = True
            
            #Slowly stop movement when keys are no longer being held
            x_difference = frame_time - abs(misc_data['resize_end_x'])
            y_difference = frame_time - abs(misc_data['resize_end_y'])
            
            if not held_keys['x'] and  x_difference < 1.0 / offset_falloff and update_yet:
                negative_multiple = int('-1'[misc_data['resize_end_x'] > 0:])
                come_to_stop = min(1, 1.0 / offset_falloff - x_difference)
                self.draw_data.offset[0] += offset_increment * negative_multiple * pow(self.draw_data.length, offset_exponential) * come_to_stop
                changed_something = True
                
            if not held_keys['y'] and  y_difference < 1.0 / offset_falloff and update_yet:
                negative_multiple = int('-1'[misc_data['resize_end_y'] > 0:])
                come_to_stop = min(1, 1.0 / offset_falloff - y_difference)
                self.draw_data.offset[1] += offset_increment * negative_multiple * pow(self.draw_data.length, offset_exponential) * come_to_stop
                changed_something = True
                    
                    
            if changed_something:
                game_flags['recalculate'] = True
                self._set_fps(self.FPS_SMOOTH)
            
            
            #Recalculate the data to draw the grid
            if game_flags['recalculate']:
            
                self._set_fps(self.FPS_MAIN)
                    
                self.draw_data.segments = self.C3DObject.segments
                self.draw_data.length = max(pow(1 / length_increment, 2) * self.draw_data.segments, 
                                            self.draw_data.length, 
                                            2)
                self.draw_data.length = float(min(2000 * self.draw_data.segments, self.draw_data.length))
                self.draw_data.angle = float(max(angle_increment, min(89, self.draw_data.angle, angle_max)))
                
                self.draw_data._calculate()
                if game_flags['reset']:
                    continue
                    
                
            #Update mouse information
            if game_flags['mouse_used'] or game_flags['recalculate']:
            
                self._set_fps(self.FPS_MAIN)
                    
                mouse_data = pygame.mouse.get_pos()
                x, y = self.to_pygame(*mouse_data)
                block_data['id'] = self.draw_data.game_to_block_index(x, y)
                block_data['taken'] = True
                if block_data['id'] is not None and ai_turn is None:
                    block_data['taken'] = self.C3DObject.grid_data[block_data['id']] != ''
            
            
            #If mouse was clicked
            if not game_flags['disable_background_clicks']:
                if (game_flags['clicked'] == 1 and not block_data['taken']) or ai_turn is not None:
                    game_flags['pending_move'] = [self.player, ai_turn if ai_turn is not None else block_data['id']]
                    misc_data['pending_move_start'] = frame_time + moving_wait
                    self._next_player()
                    continue
                   
                   
            #Highlight square
            if not block_data['taken'] and not game_flags['pending_move'] and not game_data['overlay'] and block_data['id'] is not None:
                self.C3DObject.grid_data[block_data['id']] = self.overlay_marker
                game_flags['hover'] = block_data['id']
                
                
                
            self.screen.fill(background_colour)
                
            #Draw coloured squares
            for i in self.C3DObject.range_data:
                if self.C3DObject.grid_data[i] != '':
                
                    chunk = i / self.C3DObject.segments_squared
                    coordinate = list(self.draw_data.relative_coordinates[i % self.C3DObject.segments_squared])
                    coordinate[0] += self.draw_data.offset[0]
                    coordinate[1] += self.draw_data.offset[1] - chunk * self.draw_data.chunk_height
                    
                    square = [coordinate,
                              (coordinate[0] + self.draw_data.size_x_sm,
                               coordinate[1] - self.draw_data.size_y_sm),
                              (coordinate[0],
                               coordinate[1] - self.draw_data.size_y_sm * 2),
                              (coordinate[0] - self.draw_data.size_x_sm,
                               coordinate[1] - self.draw_data.size_y_sm),
                              coordinate]
                  
                    #Player has mouse over square
                    block_colour = None
                    if self.C3DObject.grid_data[i] == self.overlay_marker:
                    
                        if game_data['players'][self.player] is False:
                            block_colour = mix_colour(WHITE, WHITE, self.player_colours[self.player])
                    
                    #Square is taken by a player
                    else:
                        j = self.C3DObject.grid_data[i]
                        
                        #Square is being moved into, mix with red and white
                        mix = False
                        if isinstance(j, int) and j > 1:
                            j = 9 - j
                            moving_block = square
                            mix = True
                        
                        block_colour = self.player_colours[j]
                        
                        if mix:
                            block_colour = mix_colour(block_colour, GREY)
                    
                    if block_colour is not None:
                        pygame.draw.polygon(self.screen,
                                            block_colour,
                                            [self.to_canvas(*corner)
                                             for corner in square],
                                            0)
                                        
                
            #Draw grid
            for line in self.draw_data.line_coordinates:
                pygame.draw.aaline(self.screen,
                                   BLACK,
                                   self.to_canvas(*line[0]),
                                   self.to_canvas(*line[1]),
                                   1)
            
            
            self._draw_score(players=game_data['players'],
                             winner=game_flags['winner'], 
                             pending_move=game_flags['pending_move'], 
                             flipped=game_flags['flipped'])
            
            
            if game_data['debug']:
                self._draw_debug(block_data['id'], held_keys)
            
            if game_data['overlay']:
                
                self._set_fps(self.FPS_MAIN)
                header_PADDING_TEXT = self.PADDING_TEXT[1] * 5
                subheader_PADDING_TEXT = self.PADDING_TEXT[1] * 3
                self.blit_list = []
                self.rect_list = []
                self.screen.blit(self.backdrop, (0, 0))
                screen_width_offset = (self.width - self.OVERLAY_WIDTH) / 2
                
                current_height = header_PADDING_TEXT + self.PADDING_TEXT[1]
                
                #Set page titles
                if game_data['overlay'] == 'instructions':
                    title_message = 'Instructions/About'
                    subtitle_message = ''
                elif game_data['overlay'] == 'gameend':
                    subtitle_message = 'Want to play again?'
                        
                    if game_flags['winner'] is not None and len(game_flags['winner']) == 1:
                        title_message = '{} was the winner!'.format(self.player_names[game_flags['winner'][0]])
                    else:
                        title_message = 'It was a draw!'
                        
                elif game_data['move_number'] + bool(game_flags['pending_move']) and game_data['overlay'] == 'options':
                    title_message = 'Options'
                    subtitle_message = ''
                else:
                    title_message = 'Connect 3D'
                    subtitle_message = 'By Peter Hunt'
                    
                title_text = self.font_lg.render(title_message, 1, BLACK)
                title_size = title_text.get_rect()[2:]
                self.blit_list.append((title_text, (self.PADDING_TEXT[0] + screen_width_offset, current_height)))
                
                current_height += self.PADDING_TEXT[1] + title_size[1]
                
                subtitle_text = self.font_md.render(subtitle_message, 1, BLACK)
                subtitle_size = subtitle_text.get_rect()[2:]
                self.blit_list.append((subtitle_text, (self.PADDING_TEXT[0] + screen_width_offset, current_height)))
                
                current_height += subtitle_size[1]
                if subtitle_message:
                    current_height += header_PADDING_TEXT
                    
                
                if game_data['overlay'] in ('options', 'gameend'):
                    
                    #Determine which style menu to use
                    in_progress = (game_data['move_number'] or game_flags['pending_move']) and game_data['overlay'] != 'gameend'
                    
                    #Player options
                    players_unsaved = [p1_player, p2_player]
                    players_original = list(game_data['players'])
                    player_hover = mouse_hover['player']
                    mouse_hover['player'] = None
                    options = ['Human', 'Beginner', 'Easy', 'Medium', 'Hard', 'Extreme']
                    
                    for player in range(len(game_data['players'])):
                        if players_unsaved[player] is False:
                            players_unsaved[player] = -1
                        else:
                            players_unsaved[player] = get_bot_difficulty(players_unsaved[player], _debug=True)
                        if players_original[player] is False:
                            players_original[player] = -1
                        else:
                            players_original[player] = get_bot_difficulty(players_original[player], _debug=True)
                            
                        params = []
                        for i in range(len(options)):
                            params.append([i == players_unsaved[player] or players_unsaved[player] < 0 and not i,
                                           i == players_original[player] or players_original[player] < 0 and not i,
                                           [player, i] == player_hover])
                        
                        option_data = self._draw_options('{}: '.format(self.player_names[player]),
                                                         options,
                                                         params,
                                                         screen_width_offset,
                                                         current_height)
                        
                        selected_option, options_size = option_data
                        
                        current_height += options_size
                        if not player:
                            current_height += self.PADDING_TEXT[1]
                        else:
                            current_height += subheader_PADDING_TEXT
                        
                        #Calculate mouse info
                        if selected_option is not None:
                            player_set = selected_option - 1
                            if player_set < 0:
                                player_set = False
                            mouse_hover['player'] = [player, selected_option]
                            if game_flags['clicked']:
                                if not player:
                                    p1_player = player_set
                                else:
                                    p2_player = player_set
                                if not in_progress:
                                    game_data['players'] = (p1_player, p2_player)  
                                 
                    
                    #Ask whether to flip the grid
                    options = ['Mirror/Rotate', 'Mirror', 'No']
                    option_len = len(options)
                    params = []
                    for i in range(option_len):
                        params.append([i == option_len - shuffle_level - 1,
                                       i == option_len - game_data['shuffle'][0] - 1,
                                       mouse_hover['shuffle'] is not None and i == option_len - mouse_hover['shuffle'] - 1])
                    
                    option_data = self._draw_options('Shuffle grid every 3 goes? ',
                                                     options,
                                                     params,
                                                     screen_width_offset,
                                                     current_height)
                    selected_option, options_size = option_data
                    current_height += self.PADDING_TEXT[1] + options_size
                    
                    #Calculate mouse info
                    mouse_hover['shuffle'] = None
                    if selected_option is not None:
                        mouse_hover['shuffle'] = option_len - selected_option - 1
                        if game_flags['clicked']:
                            shuffle_level = option_len - selected_option - 1
                            if not in_progress:
                                game_data['shuffle'][0] = shuffle_level
                                
                    
                    #Ask to end when there are no points left
                    options = ['Yes', 'No']
                    params = []
                    for i in range(option_len):
                        params.append([i != end_when_no_points_left,
                                       i != end_when_no_points_left,
                                       mouse_hover['end_early'] is not None and i != mouse_hover['end_early']])

                    option_data = self._draw_options('End with no available points left? ',
                                                     options,
                                                     params,
                                                     screen_width_offset,
                                                     current_height)
                    selected_option, options_size = option_data
                    current_height += subheader_PADDING_TEXT + options_size
                    
                    mouse_hover['end_early'] = None
                    if selected_option is not None:
                        mouse_hover['end_early'] = not selected_option
                        if game_flags['clicked']:
                            end_when_no_points_left = not selected_option
                    
                                
                    #Toggle hidden debug options with shift+alt+q
                    if not (not key[pygame.K_q] 
                            or not (key[pygame.K_RSHIFT] or key[pygame.K_LSHIFT])
                            or not (key[pygame.K_RALT] or key[pygame.K_LALT])):
                        
                        #Change number of segments
                        options = ['+', '-', 'Default', '({})'.format(segments)]
                        option_len = len(options)
                        options = options[:option_len - (segments == self.C3DObject.segments)]
                        params = []
                        for i in range(len(options)):
                            
                            if i == option_len - 1:
                                params.append([False, False, mouse_hover['segments'] in (0, 1)])
                            elif i == 2:
                                default_segments = segments == self.C3DObject.default_segments
                                params.append([False, default_segments, i == mouse_hover['segments']])
                            else:
                                params.append([False,
                                               False,
                                               i == mouse_hover['segments']])
                        
                        option_data = self._draw_options('Segments ({}): '.format(self.C3DObject.segments),
                                                         options,
                                                         params,
                                                         screen_width_offset,
                                                         current_height)
                        selected_option, options_size = option_data
                        
                        mouse_hover['segments'] = None
                        if selected_option != option_len - 1:
                            mouse_hover['segments'] = selected_option
                            
                            if game_flags['clicked']:
                                if selected_option in (0, 1):
                                    segments += 1 - 2 * selected_option
                                    segments = max(1, segments)
                                elif selected_option == 2:
                                    segments = 4
                                if not in_progress and self.C3DObject.segments != segments:
                                    self.C3DObject.segments = segments
                                    game_flags['reset'] = True
                        
                        current_height += self.PADDING_TEXT[1] + options_size
                        
                        
                        #Debug info
                        options = ['Yes', 'No']
                        params = []
                        for i in range(len(options)):
                            params.append([not i and game_data['debug'] or i and not game_data['debug'],
                                           not i and game_data['debug'] or i and not game_data['debug'],
                                           not i and mouse_hover['debug'] or i and not mouse_hover['debug'] and mouse_hover['debug'] is not None])
                        
                        option_data = self._draw_options('Show debug info? ',
                                                         options,
                                                         params,
                                                         screen_width_offset,
                                                         current_height)
                                                 
                        selected_option, options_size = option_data
                        
                        mouse_hover['debug'] = None
                        if selected_option is not None:
                            mouse_hover['debug'] = not selected_option
                            if game_flags['clicked']:
                                game_data['debug'] = not selected_option
                        
                        current_height += subheader_PADDING_TEXT + options_size
                    
                    
                    box_spacing = (header_PADDING_TEXT + self.PADDING_TEXT[1]) if in_progress else (self.PADDING_TEXT[1] + self.font_lg_size)
                    
                    box_height = [current_height]
                    
                    #Tell to restart game
                    if in_progress:
                        current_height += box_spacing
                        restart_message = 'Restart game to apply settings.'
                        restart_text = self.font_md.render(restart_message, 1, BLACK)
                        restart_size = restart_text.get_rect()[2:]
                        self.blit_list.append((restart_text, ((self.width - restart_size[0]) / 2, current_height)))
                        current_height += header_PADDING_TEXT
                        
                        #Continue button
                        if self._draw_button('Continue', 
                                               mouse_hover['continue'], 
                                               current_height, 
                                               -1):
                            mouse_hover['continue'] = True
                            if game_flags['clicked']:
                                game_data['overlay'] = None
                        else:
                            mouse_hover['continue'] = False
                            
                    
                    box_height.append(current_height)
                    current_height += box_spacing
                    
                    #Instructions button
                    if self._draw_button('Instructions' if in_progress else 'Help', 
                                           mouse_hover['instructions'], 
                                           box_height[0],
                                           0 if in_progress else 1):
                        mouse_hover['instructions'] = True
                        if game_flags['clicked']:
                            game_data['overlay'] = 'instructions'
                    else:
                        mouse_hover['instructions'] = False
                    
                    #New game button
                    if self._draw_button('New Game' if in_progress else 'Start', 
                                           mouse_hover['new'], 
                                           box_height[bool(in_progress)], 
                                           1 if in_progress else -1):
                        mouse_hover['new'] = True
                        if game_flags['clicked']:
                            game_flags['reset'] = True
                            game_data['overlay'] = None
                    else:
                        mouse_hover['new'] = False
                                                
                        
                    #Quit button
                    if self._draw_button('Quit to Desktop' if in_progress else 'Quit',
                                           mouse_hover['quit'], 
                                           current_height):
                        mouse_hover['quit'] = True
                        if game_flags['clicked']:
                            game_flags['quit'] = True
                    else:
                        mouse_hover['quit'] = False
                        
                #Draw background
                background_square = (screen_width_offset, header_PADDING_TEXT, self.OVERLAY_WIDTH, current_height + self.PADDING_TEXT[1] * 2)
                pygame.draw.rect(self.screen, WHITE, background_square, 0)
                pygame.draw.rect(self.screen, BLACK, background_square, 1)
                
                for i in self.rect_list:
                    rect_data = [self.screen] + i
                    pygame.draw.rect(*rect_data)
                
                for i in self.blit_list:
                    self.screen.blit(*i)
            
            
            pygame.display.flip()
                       
    def _draw_button(self, message, hover, current_height, width_multipler=0):
                       
        multiplier = 3
        
        #Set up text
        text_colour = BLACK if hover else GREY
        text_object = self.font_lg.render(message, 1, text_colour)
        text_size = text_object.get_rect()[2:]
        
        
        centre_offset = 64 * width_multipler
        text_x = (self.width - text_size[0]) / 2
        if width_multipler > 0:
            text_x += text_size[0] / 2
        if width_multipler < 0:
            text_x -= text_size[0] / 2
        text_x += centre_offset
        
        
        text_square = (text_x - self.PADDING_OPTIONS * (multiplier + 1),
                       current_height - self.PADDING_OPTIONS * multiplier,
                       text_size[0] + self.PADDING_OPTIONS * (2 * multiplier + 2),
                       text_size[1] + self.PADDING_OPTIONS * (2 * multiplier - 1))
    
        self.blit_list.append((text_object, (text_x, current_height)))
        
        #Detect if mouse is over it
        x, y = pygame.mouse.get_pos()
        in_x = text_square[0] < x < text_square[0] + text_square[2]
        in_y = text_square[1] < y < text_square[1] + text_square[3]
            
        if in_x and in_y:
            return True
                
        return False
        
        
    def _draw_options(self, message, options, params, screen_width_offset, current_height):
        """Draw a list of options and check for inputs.
        
        Parameters:
            message (str): Text to display next to the options.
            
            options (list): Names of the options.
            
            params (list): Contains information on the options.
                It needs to have the same amount of records as
                options, with each of these being a list of 3 items.
                These are used to colour the text in the correct
                way.
                
                param[option][0] = new selection
                param[option][1] = currently active
                param[option][2] = mouse hoving over
            
            screen_width_offset (int): The X position to draw the
                text.
            
            current_height (int/float): The Y position to draw the
                text.
        """
        message_text = self.font_md.render(message, 1, BLACK)
        message_size = message_text.get_rect()[2:]
        self.blit_list.append((message_text, (self.PADDING_TEXT[0] + screen_width_offset, current_height)))
        
        option_text = [self.font_md.render(i, 1, BLACK) for i in options]
        option_size = [i.get_rect()[2:] for i in option_text]
        option_square_list = []
    
        for i in range(len(options)):
            width_offset = (sum(j[0] + 2 for j in option_size[:i])
                            + self.PADDING_TEXT[0] * (i + 1) #gap between the start
                            + message_size[0] + screen_width_offset)

            option_square = (width_offset - self.PADDING_OPTIONS,
                             current_height - self.PADDING_OPTIONS,
                             option_size[i][0] + self.PADDING_OPTIONS * 2,
                             option_size[i][1] + self.PADDING_OPTIONS)
            option_square_list.append(option_square)
            
            
            #Set colours
            option_colours = list(SELECTION['Default'])
            param_order = ('Waiting', 'Selected', 'Hover')
            for j in range(len(params[i])):
                if params[i][j]:
                    rect_colour, text_colour = list(SELECTION[param_order[j]])
                    if rect_colour is not None:
                        option_colours[0] = rect_colour
                    if text_colour is not None:
                        option_colours[1] = text_colour
                    
            rect_colour, text_colour = option_colours
            
            self.rect_list.append([rect_colour, option_square])
            self.blit_list.append((self.font_md.render(options[i], 1, text_colour), (width_offset, current_height)))
        
        x, y = pygame.mouse.get_pos()
        selected_square = None
        for square in range(len(option_square_list)):
            option_square = option_square_list[square]
            in_x = option_square[0] < x < option_square[0] + option_square[2]
            in_y = option_square[1] < y < option_square[1] + option_square[3]
            if in_x and in_y:
                selected_square = square
                
        return (selected_square, message_size[1]) 
        
            
    def _format_output(self, text):
        """Format text to remove invalid characters."""
        left_bracket = ('[', '{')
        right_bracket = (']', '}')
        for i in left_bracket:
            text = text.replace(i, '(')
        for i in right_bracket:
            text = text.replace(i, ')')
        return text
    
    def _draw_score(self, players, winner, pending_move=False, flipped=False):
        """Draw the title."""
        
        #Format scores
        point_marker = '/'
        p0_points = self.C3DObject.current_points[0]
        p1_points = self.C3DObject.current_points[1]
        
        p0_font_top = self.font_md.render('{}'.format(self.player_names[0]), 1,  BLACK, self.player_colours[0])
        p1_font_top = self.font_md.render('{}'.format(self.player_names[1]), 1, BLACK, self.player_colours[1])
        p0_font_bottom = self.font_lg.render(point_marker * p0_points, 1,  BLACK)
        p1_font_bottom = self.font_lg.render(point_marker * p1_points, 1,  BLACK)
        
        p_size_top = p1_font_top.get_rect()[2:]
        p_size_bottom = p1_font_bottom.get_rect()[2:]
        
        if winner is None:
            if pending_move:
                go_message = '{} is moving...'.format(self.player_names[pending_move[0]])
            elif players[self.player] is not False:
                go_message = '{} is thinking...'.format(self.player_names[self.player])
            else:
                go_message = "{}'s turn!".format(self.player_names[self.player])
        else:
            if len(winner) != 1:
                go_message = 'The game was a draw!'
            else:
                go_message = '{} won!'.format(self.player_names[winner[0]])
            
        go_font = self.font_lg.render(go_message, 1, BLACK)
        go_size = go_font.get_rect()[2:]
        
        
        
        self.screen.blit(go_font, ((self.width - go_size[0]) / 2, self.PADDING_TEXT[1] * 3))
        self.screen.blit(p0_font_top, (self.PADDING_TEXT[0], self.PADDING_TEXT[1]))
        self.screen.blit(p1_font_top, (self.width - p_size_top[0] - self.PADDING_TEXT[0], self.PADDING_TEXT[1]))
        self.screen.blit(p0_font_bottom, (self.PADDING_TEXT[0], self.PADDING_TEXT[1] + p_size_top[1]))
        self.screen.blit(p1_font_bottom, (self.width - p_size_bottom[0] - self.PADDING_TEXT[0], self.PADDING_TEXT[1] + p_size_top[1]))

        if flipped:
            flipped_message = 'Grid was flipped!'
            flipped_font = self.font_md.render(flipped_message, 1, (0, 0, 0))
            flipped_size = flipped_font.get_rect()[2:]
            self.screen.blit(flipped_font,
                             ((self.width - flipped_size[0]) / 2,
                              self.PADDING_TEXT[1] * 3 + go_size[1]))
    
    def _draw_debug(self, block_id=None, held_keys=None):
        """Show the debug information."""
    
        mouse_data = pygame.mouse.get_pos()
        x, y = self.to_pygame(*mouse_data)
            
        info = ['DEBUG INFO',
                'FPS: {}'.format(int(round(self.clock.get_fps(), 0))),
                'Desired FPS: {}'.format(self._fps or self.FPS_IDLE),
                'Segments: {}'.format(self.C3DObject.segments),
                'Angle: {}'.format(self.draw_data.angle),
                'Offset: {}'.format(map(int, self.draw_data.offset)),
                'Side length: {}'.format(int(self.draw_data.length)),
                'Coordinates: {}'.format(mouse_data),
                'Block ID: {}'.format(block_id),
                'Keys: {}'.format(held_keys)]
                
        font_render = [self.font_sm.render(self._format_output(i), 1, BLACK) for i in info]
        font_size = [i.get_rect()[2:] for i in font_render]
        for i in range(len(info)):
            message_height = self.height - sum(j[1] for j in font_size[i:])
            self.screen.blit(font_render[i], (0, message_height))
            
        
        #Format the AI text output
        ai_message = []
        for i in self.C3DObject.ai_message:
        
            #Split into chunks of 50 if longer
            message_len = len(i)
            message = [self._format_output(i[n * 50:(n + 1) * 50]) for n in range(round_up(message_len / 50.0))]
            ai_message += message
        
        font_render = [self.font_sm.render(i, 1, BLACK) for i in ai_message]
        font_size = [i.get_rect()[2:] for i in font_render]

        for i in range(len(ai_message)):
            message_height = self.height - sum(j[1] for j in font_size[i:])
            self.screen.blit(font_render[i], (self.width - font_size[i][0], message_height))
    
def round_up(x):
    return int(x) + bool(x % 1)

if __name__ == '__main__':
    Connect3D().play()
