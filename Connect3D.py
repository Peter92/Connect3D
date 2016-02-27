from __future__ import division
from collections import defaultdict, Counter
from operator import itemgetter
from threading import Thread
import random
import base64
import zlib
import math
import cPickle
import time
import json
import urllib2
try:
    import pygame
except ImportError:
    pygame = None
VERSION = '2.0.1'
    
BACKGROUND = (252, 252, 255)
LIGHTBLUE = (86, 190, 255)
LIGHTGREY = (200, 200, 200)
GREY = (128, 128, 128)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PINK = (255, 120, 235)
PURPLE = (190, 86, 255)

SELECTION = {'Default': [WHITE, LIGHTGREY],
             'Background': [True, None],             
             'Foreground': [None, BLACK]}
             
class GameTime(object):
    def __init__(self, desired_fps=120, desired_ticks=60):
        self.start_time = time.time()
        self.desired_fps = desired_fps
        self.desired_ticks = desired_ticks
        
        self.ticks = 0
        
        self.framerate_counter = 1
        self.framerate_time = 0
        
    def calculate_ticks(self, current_time):
        """Ensure the correct number of ticks have passed since the 
        start of the game.
        This doesn't use the inbuilt pygame ticks.
        """
        time_elapsed = current_time - self.start_time
        
        total_ticks_needed = int(time_elapsed * self.desired_ticks)
        
        ticks_this_frame = total_ticks_needed - self.ticks
        self.ticks += ticks_this_frame
        
        return ticks_this_frame
    
    def calculate_fps(self, current_time, update_time=0.1):
        """Calculate the FPS from actual time, not ticks.
        
        It will return a number every update_time seconds, and will
        return None any other time.
        """
        frame_time = current_time - self.framerate_time
        
        if frame_time < update_time:
            self.framerate_counter += 1
            
        else:
            self.framerate_time = current_time
            fps = self.framerate_counter / frame_time
            self.framerate_counter = 1
            return int(fps)
        
    def limit_fps(self, alternate_fps=None):
        
        wanted_fps = alternate_fps or self.desired_fps
        if wanted_fps:
            pygame.time.Clock().tick(wanted_fps)
         
         
class GameTimeLoop(object):
    """This gets called every loop to get the relevant information from GameTime."""
    
    def __init__(self, GTObject):
    
        self.GTObject = GTObject
        GTObject.loop_start = time.time()
        
        #Run the code once so the result can be called multiple times
        self.ticks = GTObject.calculate_ticks(GTObject.loop_start)
        self.total_ticks = GTObject.ticks
        self.fps = GTObject.calculate_fps(GTObject.loop_start)
        
        self._temp_fps = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.GTObject.limit_fps(self._temp_fps)
        self._temp_fps = None
        
    def set_fps(self, fps):
        """Set a new desirect framerate."""
        self.GTObject.desired_fps = fps
    
    def temp_fps(self, fps):
        """Set a new desirect framerate just for one frame."""
        self._temp_fps = fps
    
    def update_ticks(self, ticks):
        """Change the tick rate."""
        '''
        self.GTObject.start_time = time.time()
        self.GTObject.desired_ticks = ticks
        self.ticks = 0
        '''
        #New attempt, needs testing
        self.GTObject.ticks = ticks * (self.GTObject.ticks / self.GTObject.desired_ticks)
        self.GTObject.desired_ticks = ticks
        

def format_text(x):
    """Format text to remove invalid characters."""
    left_bracket = ('[', '{')
    right_bracket = (']', '}')
    for i in left_bracket:
        x = x.replace(i, '(')
    for i in right_bracket:
        x = x.replace(i, ')')
    return x
             
             
class ThreadHelper(Thread):
    """Run a function in a background thread.
    Cannot be used to return values.
    """
    def __init__(self, function, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        Thread.__init__(self)
        self.function = function

    def run(self):
        self.function(*self.args, **self.kwargs)
        
        
def mix_colour(*args):
    """Mix together multiple colours."""
    mixed_colour = [0, 0, 0]
    num_colours = len(args)
    for colour in range(3):
        mixed_colour[colour] = sum(i[colour] for i in args) / num_colours
    return mixed_colour
    
    
def get_max_keys(x):
    """Return a list of every key containing the max value.
    Needs a dictionary of integers to work correctly.
    """
    if x:
        sorted_dict = sorted(x.iteritems(), key=itemgetter(1), reverse=True)
        if sorted_dict[0][1]:
            return sorted([k for k, v in x.iteritems() if v == sorted_dict[0][1]])
    return []
    
            
def split_list(x, n):
    """Split a list by n characters."""
    return [x[i:i+n] for i in range(0, len(x), n)]
    
    
def join_list(x):
    """Convert nested lists into one single list."""
    return [j for i in x for j in i]

    
def round_up(x):
    return int(x) + bool(x % 1)

    
class Connect3D(object):
    """Class for holding the game information.
    
    Contains the grid, level of shuffle, and score.
    Printing the class results in a text based representation of the grid.
    The grid format supports up to 255 players.
    """
    DEFAULT_SIZE = 4
    DEFAULT_SHUFFLE_LEVEL = 1
    
    def __init__(self, size=None, shuffle_level=None):
        
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
        self.grid = bytearray(0 for _ in self._range_lg)
        self.flip = FlipGrid(self)
        self.directions = DirectionCalculation(self)
        self.calculate_score()

    def __repr__(self):
        output = base64.b64encode(zlib.compress(str(self.grid)))
        return "Connect3D({}).load('{}')".format(self.size, output)

    def load(self, data, update_score=True):
        try:
            grid = bytearray(zlib.decompress(base64.b64decode(data)))
            return self.set_grid(grid, update_score=update_score)
        except ValueError:
            return self.set_grid(data, update_score=update_score)
        
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
        if not level:
            return False
            
        all_flips = (self.flip.fx, self.flip.fy, self.flip.fz, 
                     self.flip.rx, self.flip.ry, self.flip.rz, self.flip.reverse)
        max_shuffles = level * 3
        shuffles = random.sample(range(max_shuffles), random.randint(0, max_shuffles - 1))
        shuffles.append(len(all_flips) - 1)
        
        #Perform shuffle
        for i in shuffles:
            self.grid, operation = all_flips[i](self.grid)
        self.grid = bytearray(self.grid)
        
        return True
        

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
        hashes = defaultdict(set)
        
        #Get the hashes
        for id in self._range_lg:
            if self.grid[id]:
                hashes[self.grid[id]].update(self._point_score(id))
        
        #Count the hashes
        for k, v in hashes.iteritems():
            self.score[k] = len(v)
        
        return self.score


    def _point_score(self, id, player=None, quick=False):
        """Find how many points are gained from a cell, and return the row hashes.
        
        Set an optional player value to force the first value to be that player.
        """
        
        if player is None:
            player = self.grid[id]
        
        if quick:
            if not player:
                return 0
        else:
            row_hashes = set()
            if not player:
                return row_hashes
        
        total = 0
        calculations = 1
        for movement, invalid in self.directions:
            count = 1
                
            if not quick:
                list_match = [id]
            
            #Search both directions
            for i in (0, 1):
                point = id
                while point not in invalid[i] and 0 <= point < self._size_cubed:
                    point += movement * (-1 if i else 1)
                    if self.grid[point] == player:
                        count += 1
                        if not quick:
                            list_match.append(point)
                    else:
                        break
                    calculations += 1
                calculations += 1
            calculations += 1
                        
            #Add a point if enough matches
            if count == self.size:
                if quick:
                    total += 1
                else:
                    row_hashes.add(hash(tuple(sorted(list_match))))
        
        if quick:
            return total, calculations
        else:
            return row_hashes
        

class Connect3DGame(object):
    DEFAULT_PLAYERS = 2
    DEFAULT_SHUFFLE_TURNS = 3
    
    def __init__(self, players, shuffle_level=None, shuffle_turns=None, size=None):
        self.shuffle_turns = self.DEFAULT_SHUFFLE_TURNS if shuffle_turns is None else max(0, shuffle_turns)
        self.core = Connect3D(size=size, shuffle_level=shuffle_level)
        self.ai = ArtificialIntelligence(self)
        self._ai_text = []
        self._ai_move = None
        self._ai_state = None
        self._ai_running = False
        
        self.players = list(players)
        for i, player in enumerate(self.players):
            if player is True:
                self.players[i] = self.ai.DEFAULT_DIFFICULTY + 1
            if not (0 <= player <= 5):
                raise ValueError('invalid player level')
        self.players = bytearray(self.players)
        
        self._player_count = len(self.players)
        self._range_players = [i + 1 for i in range(self._player_count)]
        self._player = -1
        self._player_types = [i - 1 for i in self.players]
        
    def next_player(self, player, num_players):
        player += 1
        if player != num_players:
            player %= num_players
        return player
    
    def previous_player(self, player, num_players):
        player -= 1
        if player == 0:
            player = num_players
        return player
    
    def __repr__(self):
        data = bytearray(list(self.players) + [self._player, 6]) + self.core.grid
        output = base64.b64encode(zlib.compress(str(data)))
        return "Connect3DGame.load('{}')".format(output)
        
    @classmethod
    def load(cls, data):
        """Load a grid into the game."""
        
        decoded_data = bytearray(zlib.decompress(base64.b64decode(data)))
        players, grid = decoded_data.split(chr(6), 1)
        player = players.pop(-1)
        
        cube_root = pow(len(grid), 1/3)
        
        if round(cube_root) != round(cube_root, 4):
            raise ValueError('incorrect input size')
            
        #Create new class
        new_instance = cls(players=players, size=int(round(cube_root)))
        new_instance.core.set_grid(grid)
        new_instance._player = player
        return new_instance
    
    def check_game_end(self, end_early=True):
    
            #Check if any points are left to gain
            points_left = True
            if end_early:
                potential_points = {j: Connect3D(self.core.size).set_grid([j if not i else i for i in self.core.grid]).score for j in self._range_players}
                if all(self.core.score == potential_points[player] for player in self._range_players):
                    points_left = False
                    
            #Check if no spaces are left
            if 0 not in self.core.grid or not points_left:
                return get_max_keys(self.core.score)
           
    def play(self, basic=False):
        
        if self._player == -1:
            self._player = random.choice(self._range_players)
        if pygame and not basic:
            return GameCore(self).play()
        
        max_go = self.core._size_cubed - 1
        count_shuffle = 0
        flipped = False
        while True:
            
            #Print information
            print self.core
            print 'Scores: {}'.format(dict(self.core.calculate_score()))
            if flipped:
                flipped = False
                print "Grid was flipped!"
            #if self._ai_text:
            #    print self._ai_text
            self._ai_text = []
            
            winning_player = self.check_game_end(self._range_players)
            if winning_player is not None:
                if len(winning_player) == 1:
                    print 'Player {} won!'.format(winning_player[0])
                else:
                    print 'The game was a draw!'
                    
                #Ask to play again and check if answer is a variant of 'yes' or 'ok'
                print 'Play again?'
                play_again = raw_input().lower()
                if any(i in play_again for i in ('y', 'k')):
                    self.core = Connect3D(size=self.core.size, shuffle_level=self.core.shuffle_level)
                    return self.play(basic=basic)
                else:
                    return
            
            
            print "Player {}'s turn".format(self._player)
            player_type = self._player_types[self._player - 1]
            
            #Human move
            if player_type < 0:
            
                while True:
                    new_go = raw_input()
                    if not len(new_go):
                        return
                    try:
                        new_go = int(new_go)
                    except ValueError:
                        print 'input must be an integer between 0 and {}'.format(max_go)
                        continue
                    if 0 <= new_go <= max_go and not self.core.grid[new_go]:
                        self.core.grid[new_go] = self._player
                        break
                    if new_go > max_go:
                        print 'input must be between 0 and {}'.format(max_go)
                    elif self.core.grid[new_go]:
                        print 'input is taken'
                    else:
                        print 'unknown error with input'
            
            #Computer move
            else:
                new_go = self.ai.calculate_move(self._player, difficulty=player_type, _range=self._range_players)
                self.core.grid[new_go] = self._player
            
            self._player = self.next_player(self._player, self._player_count)
            
            #Flip the grid
            count_shuffle += 1
            if count_shuffle >= self.shuffle_turns:
                count_shuffle = 0
                self.core.shuffle()
                flipped = True
             
        
def direction_calculation(C3D):
    """Calculate the directions to move in.
    This is needed as the grid is a 1D list being treated as 3D.
    
    The direction moves are the amount of movement needed to go in a
    direction. For example, going right is +1, but going up is -size^2.
    
    The edges refer to the sides of the grid in a particular direction,
    so that the code knows where to stop looking.
    
    Since the code must look both ways from a single cell, directions
    directly opposite of each other are not needed, so this works by
    building a list of every direction, and trimming it down to remove 
    any opposites.
    
    Parameters:
        C3D (Connect3D): Needed to calculate the edges and movement.
    """
    
    direction_group = {}
    direction_group['X'] = 'LR'
    direction_group['Y'] = 'UD'
    direction_group['Z'] = 'FB'
    direction_group[' '] = ' '
    
    #Calculate the edge numbers for each direction
    edges = {'U': list(C3D._range_md),
             'D': range(C3D._size_squared * (C3D.size - 1), C3D._size_cubed),
             'R': [i * C3D.size + C3D.size - 1 for i in C3D._range_md],
             'L': [i * C3D.size for i in C3D._range_md],
             'F': [i * C3D._size_squared + C3D._size_squared + j - C3D.size
                   for i in C3D._range_sm for j in C3D._range_sm],
             'B': [i * C3D._size_squared + j for i in C3D._range_sm for j in C3D._range_sm],
             ' ': []}
                  
    #Calculate the addition needed to move in each direction
    move = {'U': -C3D._size_squared,
            'D': C3D._size_squared,
            'L': -1,
            'R': 1,
            'F': C3D.size,
            'B': -C3D.size,
            ' ': 0}
    
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
    
    #Calculate actual directions specific to current grid size
    reverse_directions = []
    for direction in opposite_direction:
        
        #Get a list of directions and calculate movement amount
        directions = [list(direction)]
        directions += [[j.replace(i, '') 
                       for i in directions[0] 
                       for j in direction_group.values() 
                       if i in j]]
        direction_movement = sum(move[j] for j in directions[0])
                        
        #Build list of invalid directions
        invalid_directions = [[edges[j] for j in directions[k]] for k in (0, 1)]
        invalid_directions = [join_list(j) for j in invalid_directions]
        
        reverse_directions.append((direction_movement, invalid_directions))
    
    return reverse_directions
        
    
class FlipGrid(object):
    """Use the size of the grid to calculate how flip it on the X, Y, or Z axis.
    The flips keep the grid intact but change the perspective of the game.
    """
    def __init__(self, C3D):
        self.C3D = C3D
    
    def fx(self, data):
        """Flip on the X axis."""
        return join_list(x[::-1] for x in split_list(data, self.C3D.size)), 'fx'
        
    def fy(self, data):
        """Flip on the Y axis."""
        return join_list(join_list(split_list(x, self.C3D.size)[::-1]) 
                                   for x in split_list(data, self.C3D._size_squared)), 'fy'
        
    def fz(self, data):
        """Flip on the Z axis."""
        return join_list(split_list(data, pow(self.C3D.size, 2))[::-1]), 'fz'
    
    def rx(self, data, reverse=None):
        """Rotate on the X axis."""
        if reverse is None:
            reverse = random.randint(0, 1)
        
        start = self.C3D._size_cubed - self.C3D._size_squared
        if reverse:
            return [data[start + i + j * self.C3D.size - k * self.C3D._size_squared] 
                    for i in self.C3D._range_sm_r 
                    for j in self.C3D._range_sm 
                    for k in self.C3D._range_sm_r], 'rx1'
        else:
            return [data[start + i + j * self.C3D.size - k * self.C3D._size_squared] 
                    for i in self.C3D._range_sm 
                    for j in self.C3D._range_sm 
                    for k in self.C3D._range_sm], 'rx2'
            
    def ry(self, data, reverse=None):
        """Rotate on the Y axis."""
        if reverse is None:
            reverse = random.randint(0, 1)
        
        split = split_list(data, self.C3D._size_squared)
        if reverse:
            return join_list(j[offset:offset + self.C3D.size] 
                             for offset in [(self.C3D.size - i - 1) * self.C3D.size 
                                            for i in self.C3D._range_sm]
                             for j in split), 'ry1'
        else:
            split = split[::-1]
            return join_list(j[offset:offset + self.C3D.size] 
                             for offset in [i * self.C3D.size 
                                            for i in self.C3D._range_sm] 
                             for j in split), 'ry2'
            
    def rz(self, data, reverse=None):
        """Rotate on the Z axis."""
        if reverse is None:
            reverse = random.randint(0, 1)
            
        split = split_list(data, self.C3D._size_squared)
        if reverse:
            return [x[j][i] 
                    for x in [split_list(x, self.C3D.size) 
                              for x in split] 
                    for i in self.C3D._range_sm_r for j in self.C3D._range_sm], 'rz1'
        else:
            return [x[j][i] 
                    for x in [split_list(x, self.C3D.size)[::-1] 
                              for x in split] 
                    for i in self.C3D._range_sm for j in self.C3D._range_sm], 'rz2'
    
    def reverse(self, data):
        """Reverse the grid."""
        return data[::-1], 'r'


class ArtificialIntelligence(object):
    """AI coded to play Connect3D."""
    
    DEFAULT_DIFFICULTY = 2
    
    def __init__(self, C3DGame):
        self.game = C3DGame
        self._temp_core = Connect3D(self.game.core.size)
    

    def check_cell(self, cell_id, grid, player=None):
        """Check how many points a cell has for a specific grid.
        
        Parameters:
            grid (list/tuple): 1D list of grid cells, amount must be a cube number.
            
            cell_id (int): The cell ID, or grid_data index to update.
            
            player (int): Integer representation of the player, can be 0 or 1.
        """
        
        self._temp_core.grid = grid
        total, calculations = self._temp_core._point_score(cell_id, player, quick=True)
        try:
            self.calculations += calculations
        except AttributeError:
            pass
        return total
        
        
    def points_bestcell(self, player):
        """Get maximum number of points that can be gained from each empty cell,
        that is not blocked by an enemy value.
        """
        max_points = defaultdict(int)
        filled_grid = bytearray(i if i else player for i in self.game.core.grid)
        for cell_id in self.game.core._range_lg:
            if filled_grid[cell_id] == player and not self.game.core.grid[cell_id]:
                max_points[cell_id] = self.check_cell(cell_id, filled_grid)
        
        return get_max_keys(max_points)

    def points_immediateneighbour(self, player_range, grid=None):
        """Find all places where anyone has n-1 points in a row, by substituting
        in a point for each player in every cell.
        
        Parameters:
            grid_data (list or None, optional): Pass in a custom grid_data, 
                leave as None to use the Connect3D one.
        """
        if grid is None:
            grid = bytearray(self.game.core.grid)
        
        matches = defaultdict(list)
        for cell_id in self.game.core._range_lg:
            if not grid[cell_id]:
                for player in player_range:
                    if self.check_cell(cell_id, grid, player):
                        matches[player].append(cell_id)
        
        return matches
    
    def points_nearneighbour(self, player_range):
        """Look two moves ahead to detect if someone could get a point.
        Uses the check_for_n_minus_one function from within a loop.
        
        Will return 1 as the second parameter if it has looked up more than a single move.
        """
        
        #Initial check
        initial_match = self.points_immediateneighbour(player_range=player_range)
        match_cells = []
        
        #Make list of all cells so far to avoid duplicates
        for k, v in initial_match.iteritems():
            match_cells += v
        match_cells = set(match_cells)
            
        #For every grid cell, substitute a player into it, then do the check again
        grid = bytearray(self.game.core.grid)
        matches = defaultdict(list)
        for i in self.game.core._range_lg:
            if not self.game.core.grid[i]:
                old_value = grid[i]
                for player in player_range:
                    grid[i] = player
                    match = self.points_immediateneighbour(player_range=[player], grid=grid)
                    if match:
                        for k, v in match.iteritems():
                            matches[k] += [cell for cell in v if cell not in match_cells or v.count(cell) > 1]
                            
                grid[i] = old_value
        
        return initial_match, matches
        

    def calculate_move(self, player, difficulty=None, _range=None):
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
        self.game._ai_running = True
        
        if _range is None:
            _range = (1, 2)
        
        chance_tactic, chance_ignore, chance_ignore_offset = self.difficulty(difficulty)
        
        total_moves = len([i for i in self.game.core.grid if i])
        self.calculations = 0
        next_moves = []
        
        self.game._ai_text = ['AI Objective: Calculating']
        self.game._ai_state = None
        self.game._ai_move = None
        ai_text = self.game._ai_text.append
        
        #It is possible the first few moves since they need the most calculations
        #This is disabled for now though as the AI runs a lot faster
        non_dangerous_skip = total_moves >= (self.game.core.size - 2) * len(_range)
        
        if True:
            
            #Calculate move
            close_matches, far_matches = self.points_nearneighbour(player_range=_range)
            del self.game._ai_text[0]
            ai_text('Urgent: {}'.format(bool(close_matches)))
            
            #Reduce chance of not noticing n-1 in a row, since n-2 in a row isn't too important
            chance_ignore_near = 0
            if close_matches:
                chance_ignore_near = pow(chance_ignore / chance_ignore_offset, 
                                         pow(total_moves / self.game.core._size_cubed, 0.25))
            
            #Chance of things happening
            chance_notice_basic = random.uniform(0, 100) > chance_ignore_near
            chance_notice_far = random.uniform(0, 100) > chance_ignore
            chance_notice_advanced = min(random.uniform(0, 100), random.uniform(0, 100)) > chance_ignore
            
            #Count occurances, set overall total to [0]
            move_count_player = defaultdict(dict)
            move_count_player[0] = defaultdict(int)
            move_count_advanced = defaultdict(list)
            for k, v in far_matches.iteritems():
                if k not in move_count_player:
                    move_count_player[k] = defaultdict(int)
                for i in v:
                    move_count_player[0][i] += 1
                    move_count_player[k][i] += 1
                for k2, v2 in move_count_player[k].iteritems():
                    if v2 > 1:
                        move_count_advanced[k] += [k2] * (v2 - 1)
            
            advanced_move = any(v > 1 for k, v in move_count_player[0].iteritems())
            
            #First try block an enemy advanced move, then do own
            #Then do the move that would block an enemy and gain a point at the same time
            if advanced_move and chance_notice_advanced:
                next_moves_total = move_count_advanced[0]
                next_moves_player = move_count_advanced[player]
                del move_count_advanced[player]
                del move_count_advanced[0]
                
                #Enemy moves
                for k, v in move_count_advanced.iteritems():
                    if k:
                        next_moves = move_count_advanced[k]
                        self.game._ai_state = 'Forward Thinking (Blocking Opposition)'
                
                #Own moves
                if next_moves_player:
                    if not next_moves or random.uniform(0, 100) < chance_tactic:
                        next_moves = next_moves_player
                        self.game._ai_state = 'Forward Thinking (Gaining Points)'
                    
                #Leftover moves
                if next_moves_total and not next_moves:
                    next_moves = next_moves_player
                    self.game._ai_state = 'Forward Thinking'
            
            
            #Check for any n-1 points
            #Block enemy first then gain points
            if close_matches and chance_notice_basic:
            
                already_moved = bool(next_moves)
                
                if close_matches[player]:
                    if not chance_notice_advanced or not next_moves:
                        next_moves = close_matches[player]
                        self.game._ai_state = 'Gaining Points'
            
                enemy_moves = []
                for k, v in close_matches.iteritems():
                    if k != player:
                        enemy_moves += v
                
                if enemy_moves:
                    if not next_moves or not random.uniform(0, 100) < chance_tactic:
                        next_moves = enemy_moves
                        self.game._ai_state = 'Blocking Opposition'
                        
            
            #Check for any n-2 points
            #Gain points first then block enemy
            elif not next_moves and chance_notice_far:
                next_moves = far_matches[player]
                self.game._ai_state = 'Looking Ahead (Gaining Points)'
                
                enemy_moves = []
                for k, v in far_matches.iteritems():
                    if k != player:
                        enemy_moves += v
                
                if enemy_moves:
                    if not next_moves or random.uniform(0, 100) < chance_tactic:
                        next_moves = []
                        for k, v in far_matches.iteritems():
                            next_moves += v
                        self.game._ai_state = 'Looking Ahead (Blocking Opposition)'
                
            
            if not self.game._ai_state:
                if not chance_notice_basic and not chance_notice_advanced:
                    ai_text("AI missed something.")
                self.game._ai_state = False
                
        
        #Make a semi random placement
        if (not next_moves or not self.game._ai_state) and not chance_ignore and random.uniform(0, 100) > chance_ignore:
            next_moves = self.points_bestcell(player)
            self.game._ai_state = 'Predictive placement'
            
        #Make a totally random move
        if not next_moves or not self.game._ai_state:
            next_moves = [i for i in self.game.core._range_lg if not self.game.core.grid[i]]
            self.game._ai_state = 'Random placement'
                
        ai_text('AI Objective: {}.'.format(self.game._ai_state))
        n = random.choice(next_moves)
        
        potential_moves = 'Potential Moves: {}'.format(next_moves)
        if len(potential_moves) > 40:
            potential_moves = potential_moves[:37] + '...'
        ai_text(potential_moves)
        
        self.game._ai_move = random.choice(next_moves)
        
        ai_text('Chosen Move: {}'.format(self.game._ai_move))
        ai_text('Calculations: {}'.format(self.calculations + 1))
        
        #print state, self.game._ai_text
        self.game._ai_running = False
        return self.game._ai_move
        

    def difficulty(self, level=None):
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
        if level is None:
            level = self.DEFAULT_DIFFICULTY
        
        level_data = [(75, 95, 1), #Beginner
                      (50, 75, 2), #Easy
                      (40, 50, 3), #Medium
                      (20, 25, 3),  #Hard
                      (0, 0, 1)]    #Extreme
                      
        return level_data[level]
        

#PYGAME STUFF
class DrawData(object):
    def __init__(self, C3DCore, length, angle, padding, offset):
        self.core = C3DCore
        self.length = length
        self.angle = angle
        self.padding = padding
        self.offset = offset
        self.recalculate()
    
    def recalculate(self):
        """Perform the main calculations on the values in __init__.
        This allows updating any of the values, such as the isometric
        angle, without creating a new class."""
        
        self.size_x = self.length * math.cos(math.radians(self.angle))
        self.size_y = self.length * math.sin(math.radians(self.angle))
        self.x_offset = self.size_x / self.core.size
        self.y_offset = self.size_y / self.core.size
        self.chunk_height = self.size_y * 2 + self.padding
        
        self.centre = (self.chunk_height // 2) * self.core.size - self.padding / 2
        self.size_x_sm = self.size_x / self.core.size
        self.size_y_sm = self.size_y / self.core.size
        
        self.length_small = self.length / self.core.size
        
        
        #Square
        self.relative_coordinates = []
        position = (0, self.centre)
        for j in self.core._range_sm:
            checkpoint = position
            for i in self.core._range_sm:
                self.relative_coordinates.append(position)
                position = (position[0] + self.x_offset,
                            position[1] - self.y_offset)
            position = (checkpoint[0] - self.x_offset,
                        checkpoint[1] - self.y_offset)
                        
        
        #Absolute coordinates for pygame
        #chunk_coordinates = [(self.offset[0], self.offset[1] - i * self.chunk_height) for i in self.core._range_sm]
        
        bottom_height = self.offset[1] + self.centre - self.size_y
        top_height = bottom_height + self.chunk_height * (1 - self.core.size)
        
        self.line_coordinates = [((self.offset[0] + self.size_x, bottom_height),
                                  (self.offset[0] + self.size_x, top_height)),
                                 ((self.offset[0] - self.size_x, bottom_height),
                                  (self.offset[0] - self.size_x, top_height)),
                                 ((self.offset[0], top_height + self.size_y),
                                  (self.offset[0], bottom_height + self.size_y))]
        
        bottom_height = self.offset[1] + self.centre - self.size_y
        top_height = self.offset[1] + self.centre - self.size_y * 2
        for i in self.core._range_sm:

            chunk_height = -i * self.chunk_height
            
            self.line_coordinates += [((self.offset[0] + self.size_x, bottom_height + chunk_height),
                                       (self.offset[0], top_height + chunk_height)),
                                      ((self.offset[0] - self.size_x, bottom_height + chunk_height),
                                       (self.offset[0], top_height + chunk_height))]

            for coordinate in self.relative_coordinates:
                
                start = (self.offset[0] + coordinate[0], self.offset[1] + chunk_height + coordinate[1])
                self.line_coordinates += [(start,
                                           (start[0] + self.size_x_sm, start[1] - self.size_y_sm)),
                                          (start,
                                           (start[0] - self.size_x_sm, start[1] - self.size_y_sm))]
        
        
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
        
        n = self.core.size
        if 0 <= x < n and 0 <= y < n and 0 <= z < n:
            return (y + n * (x + n * z))
        else:
            return None
            

class GameCore(object):

    FPS_IDLE = 5
    FPS_MAIN = 60
    FPS_SMOOTH = 120
    
    TICKS = 120
    WIDTH = 640
    HEIGHT = 960
    MOVE_WAIT = 60
    TIMER_DEFAULT = 200
    
    def __init__(self, C3DGame):
        self.game = C3DGame
        self.timer_count = self.TIMER_DEFAULT
        self.timer_enabled = True
        
        self.colour_order = [GREEN, YELLOW, LIGHTBLUE, PINK, PURPLE, RED]
        self.player_colours = list(self.colour_order)
        random.shuffle(self.player_colours)
    
    def __repr__(self):
        return self.game.__repr__()
    
    def exit(self):
        """Handle ending the game from anywhere."""
        self.update_state(None)

    def _new_surface(self, height, blit_list, rect_list):
        surface = pygame.Surface((self.menu_width, height + self.menu_padding))
        surface.fill(WHITE)
        
        for rect in rect_list:
            pygame.draw.rect(*([surface] + rect))
                    
        for font in blit_list:
            surface.blit(*font)
        
        return surface
        
    def _grid_surface(self, core=None, draw=None, hover=None, pending=None, early=None, background=None, player_colours=None, width=None, height=None):
    
        colours = player_colours or self.player_colours
        core = core or self.game.core
        draw = draw or self.draw
        extra = [hover, pending, early]
        try:
            extra[0] = extra[0][0]
        except TypeError:
            pass
        try:
            extra[1] = extra[1][0]
        except TypeError:
            pass
        try:
            extra[2] = extra[2][0]
        except TypeError:
            pass
            
        surface = pygame.Surface((width or self.WIDTH, height or self.HEIGHT))
        surface.fill(background if background else BACKGROUND)
        
        for i in core._range_lg:
            if core.grid[i] or i in extra:
                i_reverse = core._size_cubed - i - 1
                chunk = i_reverse // core._size_squared
                base_coordinate = draw.relative_coordinates[i_reverse % core._size_squared]
                
                coordinate = (draw.offset[0] - base_coordinate[0],
                              base_coordinate[1] + draw.offset[1] - chunk * draw.chunk_height)
                square = [coordinate,
                          (coordinate[0] + draw.size_x_sm,
                           coordinate[1] - draw.size_y_sm),
                          (coordinate[0],
                           coordinate[1] - draw.size_y_sm * 2),
                          (coordinate[0] - draw.size_x_sm,
                           coordinate[1] - draw.size_y_sm),
                          coordinate]
                          
                #Player has mouse over square
                block_colour = None
                
                if not core.grid[i]:
                    
                    #Hovering over block
                    if i == extra[0]:
                        block_colour = mix_colour(WHITE, WHITE, colours[hover[1] - 1])
                        
                    if i == extra[1]:
                        player_colour = colours[self.game._player - 1]
                        
                        #Holding down over block
                        if pending[2]:
                            block_colour = mix_colour(BLACK, GREY, WHITE, WHITE, player_colour, player_colour, player_colour, player_colour)
                        
                        #Holding down but moved away
                        else:
                            block_colour = mix_colour(WHITE, WHITE, player_colour)
                    
                    #Hovering over block between turns
                    if i == extra[2]:
                        block_colour = mix_colour(WHITE, WHITE, colours[early[1] - 1])
                
                #Square is taken by a player
                else:
                    block_colour = colours[core.grid[i] - 1]
                
                if block_colour is not None:
                    pygame.draw.polygon(surface,
                                        block_colour, square, 0)
        
              
        #Draw grid
        for line in draw.line_coordinates:
            pygame.draw.aaline(surface,
                               BLACK, line[0], line[1], 1)
    
        return surface
    
    def generate_draw_data(self, C3D, x, y, angle_range=None, width_limits=None, height_limits=None, start_offset=None):
        #Set length and angle to fit on the screen
        
        mid_point = [x // 2, y // 2]
        length = C3D.size
        
        angle_limits = angle_range or (1, 89)
        angle = angle_limits[0]
        
        offset = [mid_point[0], mid_point[1]]
        if start_offset:
            offset[0] += start_offset[0]
            offset[1] += start_offset[1]
            
        freeze_edit = False
        freeze_angle = False
        
        length_increment = length
        while True:
            edited = False
            padding = int(pow(90 - angle, 0.75) - 15)
            
            draw = DrawData(C3D, length, angle, padding, offset)
            
            height = draw.chunk_height * C3D.size
            width = draw.size_x * 2
            
            n = 0.95
            too_small = height < y * (n if not height_limits else height_limits[0])
            too_tall = height > y * (n if not height_limits else height_limits[1])
            too_thin = width < x * (n if not width_limits else width_limits[0])
            too_wide = width > x * (n if not width_limits else width_limits[1])
                    
            if too_wide or too_small and not too_thin:
                if angle < angle_limits[1]:
                    angle += 1
                    freeze_angle = True
                else:
                    length -= length_increment
                    freeze_edit = True
                edited = True
            
            if too_thin:
                if angle > angle_limits[0] and not freeze_angle:
                    angle -= 1
                    edited = True
                elif not too_tall and not freeze_edit:
                    length += length_increment
                    edited = True
                
            if too_tall:
                freeze_edit = True
                length -= length_increment
                edited = True
                
            if not edited:
                #print draw.line_coordinates
                return draw
        
    def resize_screen(self, quick=False):
        """Recalculate everything when a new width or height is set."""
        menu_width_multiplier = 20.5
        min_height = 200
        draw_width_limits = (0.85, 0.9)
        draw_height_limits = (0.85, 0.88)
        
        self.HEIGHT = max(min_height, self.HEIGHT)
        self.mid_point = [self.WIDTH // 2, self.HEIGHT // 2]
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF)
        
        self.draw = self.generate_draw_data(self.game.core, self.WIDTH, self.HEIGHT, angle_range=(24, 42),
                                          width_limits=draw_width_limits, height_limits=draw_height_limits,
                                          start_offset=(0, self.HEIGHT//25))
                
        height_multiply = min(1, 1.5 * self.WIDTH / self.HEIGHT)
        height_lg = int(round(self.HEIGHT / 28 * height_multiply))
        height_md = int(round(self.HEIGHT / 40 * height_multiply))
        height_sm = int(round(self.HEIGHT / 46 * height_multiply))
        
        #Set font sizes
        self.text_padding = max(1, self.HEIGHT // 96) * height_multiply
        self.width_padding = min(5, max(1, self.WIDTH // 128))
        self.font_lg = pygame.font.Font(self.font_file, height_lg)
        self.font_md = pygame.font.Font(self.font_file, height_md)
        self.font_sm = pygame.font.Font(self.font_file, height_sm)
        
        
        #Menu sizes
        max_width = min(640, self.WIDTH) #Width will not affect menu past this point
        self.menu_width = max(height_md, int(round(max_width / 26))) * menu_width_multiplier
        
        height_lg_m = int(round(self.menu_width / 15.5))
        height_md_m = int(round(self.menu_width / 21.5))
        height_sm_m = int(round(self.menu_width / 25))
        self.font_lg_m = pygame.font.Font(self.font_file, height_lg_m)
        self.font_md_m = pygame.font.Font(self.font_file, height_md_m)
        self.font_sm_m = pygame.font.Font(self.font_file, height_sm_m)
        self.menu_font_size = self.font_lg_m.render('', 1, BLACK).get_size()[1]
        
        self._last_title = None
        self.menu_padding = min(10, max(int(self.text_padding), max_width // 64))
        self.menu_box_padding = int(round(self.menu_width / 50))
        
        
        font_md_size = self.font_md_m.render('', 1, BLACK).get_rect()[3]
        self.menu_height_offset = self.HEIGHT // 18 * height_multiply
        self.scroll_width = self.scroll_padding = self.menu_width // 26
        
        
        #Generate example grids
        try:
            self.menu_colour
        except AttributeError:
            pass
        else:
            grid_data = [
            '                    1111                                        ',   #across
            '                 1   1   1   1                                  ',   #across
            '                   1  1  1  1                                   ',   #diagonal flat
            '                1    1    1    1                                ',   #diagonal flat
            '     1               1               1               1          ',   #down
            '    1                1                1                1        ',   #diagonal down
            '       1              1              1              1           ',   #diagonal down
            ' 1                   1                   1                   1  ',   #diagonal down
            '             1           1           1           1              ',   #diagonal down
            '1                    1                    1                    1',   #corner to corner
            '               1          1          1          1               ',   #corner to corner
            '   1                  1                  1                  1   ',   #corner to corner
            '            1            1            1            1            ']   #corner to corner
            grid_data = [map(int, list(i.replace(' ', '0'))) for i in grid_data]
            grid_data = [base64.b64encode(zlib.compress(str(bytearray(i)))) for i in grid_data]
            
            self.example_grid = []
            width = int(self.menu_width / 2.5)
            height = int(width * 1.8)
            
            for i in grid_data:
                C3D = Connect3D(4).load(i)
                draw = self.generate_draw_data(C3D, width, height)
                surface = self._grid_surface(core=C3D, draw=draw, player_colours=[self.menu_colour], width=width, height=height, background=WHITE)
                self.example_grid.append(surface)
                
            self.example_grid_count = len(self.example_grid)
                
        
        #Get menu size
        self.update_surfaces()
        menu_size = self.screen_menu_background.get_size()
        self.menu_location = [0, self.menu_height_offset]
        self.menu_location[0] = self.mid_point[0] - menu_size[0] // 2 - self.scroll_width // 2
        
    def update_surfaces(self):
        """Recalculate all surfaces and draw them to the screen."""
        self.draw_surface_main_grid()
        self.draw_surface_main_title()
        self.draw_surface_main_background()
        self.draw_surface_debug()
        self.screen.blit(self.background, (0, 0))
        self.update_state(update_state=False)
    
    def update_state(self, new_state=False, update_state=True):
        """Calculations to be done when the state of the game is changed."""
        if update_state:
            try:
                self.flag_data['LastState'] = self.state
            except AttributeError:
                pass
            
            if new_state is not False:
                self.state = new_state
        try:
            self.draw_surface_menu_settings()
            self.draw_surface_menu_instructions()
            self.draw_surface_menu_about()
            self.draw_surface_main_background()
            transparent = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA, 32)
            transparent.fill(list(WHITE) + [200])
            self.background.blit(transparent, (0, 0))
        except AttributeError:
            pass
        
        try:
            self.frame_data['Redraw'] = True
            if self.state == 'Main':
                self.frame_data['GameTime'].set_fps(self.FPS_MAIN)
            elif self.state in ('Menu', 'Instructions', 'About'):
                self.frame_data['GameTime'].set_fps(self.FPS_MAIN)
        except (AttributeError, KeyError):
            pass
    
    def draw_surface_main_background(self):
        
        grid_size = self.screen_grid.get_size()
        grid_location = [i - j / 2 for i, j in zip(self.mid_point, grid_size)]
        self.background = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.background.blit(self.screen_grid, grid_location)
        self.background.blit(self.surface_title, grid_location)
        
        try:
            time_left = self.temp_data['MoveTimeLeft']
        except AttributeError:
            time_left = None
        if time_left is not None:
            self.background.blit(self.surface_time_remaining, grid_location)
    
    def draw_surface_main_grid(self):
        """Draws the grid with coloured blocks to a surface."""
        
        try:
            hover = self.temp_data['Hover']
            pending = self.temp_data['PendingMove']
            early = self.temp_data['EarlyHover']
        except AttributeError:
            hover = pending = early = None
        
        self.screen_grid = self._grid_surface(hover=hover, pending=pending, early=early)
    
    def draw_surface_main_time(self):
        """Renders the time remaining.
        This is separate from the rest of the menu due to it needing to be redrawn a lot.
        """
        self.surface_time_remaining = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA, 32)
        try:
            time_left = self.temp_data['MoveTimeLeft']
        except AttributeError:
            time_left = None
        
        if time_left is not None and self.timer_enabled:
            time_left = int(round(time_left / self.TICKS + 0.1))
            message = '{} second{}'.format(time_left, 's' if time_left != 1 else '')
            font = self.font_sm.render(message, 1, BLACK)
            size = font.get_rect()[2:]
            self.surface_time_remaining.blit(font, ((self.WIDTH - size[0]) / 2, self.text_padding))
    
    def draw_surface_main_title(self):
        """Renders the display for the main game."""
        
        try:
            winner = self.temp_data['Winner']
            pending_move = self.temp_data['PendingMove']
            skipped = self.temp_data['Skipped']
            flipped = self.temp_data['Flipped']
        except AttributeError:
            winner = pending_move = skipped = flipped = None
            
        #Display winner
        if winner is not None:
            num_winners = len(winner)
            if len(winner) == 1:
                title = "Player {} won!".format(winner[0])
            elif 0 < num_winners < 5 and num_winners != self.game._player_count:
                winner = map(str, winner)
                numbers = '{} and {}'.format(', '.join(winner[:-1]), winner[-1])
                title = 'Players {} {} drew!'.format(numbers, 'both' if len(winner) == 2 else 'all')
            else:
                title = "The game was a draw!"
        
        elif self.game._ai_running:
            title = "Player {} is thinking...".format(self.game._player)
        
        #Don't instantly switch to player is thinking as it could be a quick click
        elif (pending_move is None
            or (not pending_move[3] and pending_move[1] > self.frame_data['GameTime'].total_ticks)):
            title = "Player {}'s turn!".format(self.game._player)
            
        else:
            if pending_move[3]:
                title = "Player {} is moving...".format(self.game._player)
            else:
                title = "Player {} is thinking...".format(self.game._player)
        
        #Hide title if starting off the game
        if self.state != 'Main' and not any(self.game.core.grid):
            title = ''
            self._last_title = None
        
        #Avoid calculating extra stuff if the move is the same
        if self._last_title == title:
            return
        
        self.surface_title = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA, 32)
        font = self.font_lg.render(title, 1, BLACK)
        main_size = font.get_rect()[2:]
        self.surface_title.blit(font, ((self.WIDTH - main_size[0]) / 2, self.text_padding * 3))
        
        #Get two highest scores (or default to player 1 and 2)
        #If matching scores, get lowest player
        score = dict(self.game.core.score)
        try:
            max_score = max(score.iteritems(), key=itemgetter(1))[1]
            player1, score1 = min(((p, s) for p, s in score.iteritems() if s == max_score), key=itemgetter(0))
            del score[player1]
            if not score1:
                raise ValueError()
        except ValueError:
            player1 = 1
            score1 = 0
        try:
            max_score = max(score.iteritems(), key=itemgetter(1))[1]
            player2, score2 = min(((p, s) for p, s in score.iteritems() if s == max_score), key=itemgetter(0))
            if not score2:
                raise ValueError()
        except ValueError:
            player2 = 2 if player1 == 1 else 1
            score2 = 0
            
        #Switch values so lowest player will come first
        if player2 < player1:
            _player = player2
            _score = score2
            player2 = player1
            score2 = score1
            player1 = _player
            score1 = _score
        
        
        #Set menu colour based on the 2 players
        for colour in self.colour_order:
            if colour in (self.player_colours[player1 - 1], self.player_colours[player2 - 1]):
                self.menu_colour = colour
                break
        

        point_display = '/'
        
        #Adjust number of points in a row to display
        if self._last_title != title:
            size_remaining = (self.WIDTH - main_size[0]) / 2
            current_size = 0
            n = 9 #Start at minimum length of scores + 4
            while current_size < size_remaining:
                current_size = self.font_lg.render(point_display * n, 1, BLACK).get_rect()[2]
                n += 1
            self.score_width = n - 4
            
        #Score 1
        upper_font = self.font_md.render('Player {}'.format(player1), 1, BLACK, self.player_colours[player1 - 1])
        upper_size = upper_font.get_rect()[2:]
        self.surface_title.blit(upper_font, (self.width_padding, self.text_padding))
        
        current_height = self.text_padding + upper_size[1]
        for i in range(score1 // self.score_width + 1):
            points = self.score_width if (i + 1) * self.score_width <= score1 else score1 % self.score_width
            lower_font = self.font_lg.render(point_display * points, 1, BLACK)
            lower_size = lower_font.get_rect()[2:]
            self.surface_title.blit(lower_font, (self.width_padding, current_height))
            current_height += lower_size[1] - self.text_padding
        
        #Score 2
        upper_font = self.font_md.render('Player {}'.format(player2), 1, BLACK, self.player_colours[player2 - 1])
        upper_size = upper_font.get_rect()[2:]
        self.surface_title.blit(upper_font, (self.WIDTH - upper_size[0] - self.width_padding, self.text_padding))
        
        current_height = self.text_padding + upper_size[1]
        for i in range(score2 // self.score_width + 1):
            points = self.score_width if (i + 1) * self.score_width <= score2 else score2 % self.score_width
            lower_font = self.font_lg.render(point_display * points, 1, BLACK)
            lower_size = lower_font.get_rect()[2:]
            self.surface_title.blit(lower_font, (self.WIDTH - lower_size[0] - self.text_padding, current_height))
            current_height += lower_size[1] - self.text_padding
        
        #Status message
        if winner is None and (skipped or flipped) and self._last_title != title:
            if skipped:
                if skipped == 2:
                    message = 'Forced move!'
                else:
                    message = 'Switched players!'
                last_player = self.game.previous_player(self.game._player, self.game._player_count)
                message += ' (Player {} took too long)'.format(last_player)
            elif flipped:
                message = 'Grid was flipped!'
            font = self.font_md.render(message, 1, (0, 0, 0))
            size = font.get_rect()[2:]
            self.surface_title.blit(font, ((self.WIDTH - size[0]) / 2, self.text_padding * 3 + main_size[1]))
        
        self._last_title = title
        #self.draw_surface_main_time()  
    
    def draw_surface_menu_container(self, update_scroll=True):
        
        start_scroll = self.option_set['Scroll']
        if self.state == 'Menu':
            contents = self.screen_menu_background
        elif self.state == 'Instructions':
            contents = self.screen_menu_instructions
        elif self.state == 'About':
            contents = self.screen_menu_about
        else:
            contents = self.screen_menu_background
        contents_size = contents.get_size()[1]
        
        menu_width = self.menu_width + self.scroll_width // 2 + 1
    
        max_height = self.HEIGHT - self.menu_location[1] * 4
        min_height = self.menu_padding
        menu_height = max(min_height, min(max_height, contents_size))
        
        
        #Mouse wheel
        used_mouse_wheel = False
        if update_scroll:
            used_mouse_wheel = self.frame_data['MouseClick'][3] or self.frame_data['MouseClick'][4]
            scroll_speed = menu_height / 10
            self.option_set['Scroll'] += scroll_speed * self.frame_data['MouseClick'][3]
            self.option_set['Scroll'] -= scroll_speed * self.frame_data['MouseClick'][4]
        
            
        #Scroll bar
        scroll_top = self.scroll_padding
        scroll_bottom = (menu_height - self.scroll_padding - scroll_top) * (menu_height / contents_size)
        
        #Set correct offset
        if self.option_hover['Scroll'] is not None:
            offset = self.option_hover['Scroll'][0] - self.option_hover['Scroll'][1]
        else:
            offset = self.option_set['Scroll']
        
        #Correctly size the scroll speed and scroll bar
        offset_adjusted = self.scroll_padding * 3 + scroll_bottom - scroll_top - menu_height
        offset = max(offset_adjusted, min(0, offset))
            
        scroll_top -= offset
        if offset_adjusted:
            offset *= (menu_height - contents_size) / offset_adjusted
            
        scroll_dimensions = [self.menu_width - self.scroll_width // 2, scroll_top, self.scroll_width, scroll_bottom]
        if update_scroll:
        
            if self.option_set['Scroll'] is not None:
                self.option_set['Scroll'] = max(offset_adjusted, min(0, self.option_set['Scroll']))
                
            try:
                x, y = self.frame_data['MousePos']
                
            except AttributeError:
                pass
                
            else:
                x -= self.menu_location[0] + self.scroll_width // 2
                y -= self.menu_location[1]
                x_selected = scroll_dimensions[0] < x < scroll_dimensions[0] + scroll_dimensions[2]
                y_selected = scroll_dimensions[1] < y < scroll_dimensions[1] + scroll_dimensions[3]
                
                if self.frame_data['MouseClick'][0]:
                    if self.option_hover['Scroll'] is None:
                        if x_selected and y_selected:
                            self.option_hover['Scroll'] = [y, y]
                        if self.option_set['Scroll'] is not None and self.option_hover['Scroll'] is not None:
                            self.option_hover['Scroll'][0] += self.option_set['Scroll']
                    else:
                        self.option_hover['Scroll'][1] = y
                    
                    self.frame_data['GameTime'].temp_fps(self.FPS_SMOOTH)
                        
                elif self.option_hover['Scroll'] is not None:
                    self.option_set['Scroll'] = self.option_hover['Scroll'][0] - self.option_hover['Scroll'][1]
                    self.option_hover['Scroll'] = None
            self.scroll_offset = offset
        
        
        self.screen_menu_holder = pygame.Surface((self.menu_width + self.scroll_width // 2 + 1, menu_height), pygame.SRCALPHA, 32)
        self.screen_menu_holder.blit(contents, (0, offset))
        
        #Draw outline
        pygame.draw.rect(self.screen_menu_holder, BLACK, (0, 0, self.menu_width, menu_height), 1)
        
        #Draw scroll bar
        pygame.draw.rect(self.screen_menu_holder, self.menu_colour, scroll_dimensions, 0)
        pygame.draw.rect(self.screen_menu_holder, BLACK, scroll_dimensions, 1)
        
        if start_scroll != self.option_set['Scroll']:
            self.frame_data['Redraw'] = True
        return used_mouse_wheel
        
    def draw_surface_menu_about(self):
    
        mouse_clicked = self._mouse_click()
        height_current = self.menu_padding
        blit_list = []
        rect_list = []
        
        title_message = 'About'
        subtitle_message = ''
        height_current = self._generate_menu_title(title_message, subtitle_message, height_current, blit_list)
        
        text_chunks = [
        'Version: {} (February 2016)'.format(VERSION),
        'Website: http://peterhuntvfx.co.uk',
        'Email: peterhuntvfx@yahoo.com',
        ]
        for message in text_chunks:
            if not message:
                height_current += self.menu_padding
                continue
            result = self._generate_menu_selection(message,
                                            [], [], height_current,
                                            blit_list, rect_list, 
                                            font=self.font_md_m)
            _, height_current = result
        height_current += self.menu_padding
        
        
        text_chunks = [
        'If you have any questions or feedback please feel free',
        'to get in touch via email.',
        '',
        '',
        'This is a much more optimised version of the game with',
        'a few extra features added and AI slightly tweaked. It ',
        'has only been tested by me so far, so you may still find',
        'a few bugs.'
        
        ]
        for message in text_chunks:
            if not message:
                height_current += self.menu_padding
                continue
            result = self._generate_menu_selection(message,
                                            [], [], height_current,
                                            blit_list, rect_list, 
                                            font=self.font_sm_m)
            _, height_current = result
            
            
        height_current += self.menu_padding * 2
        result = self._generate_menu_button('Back',
                                        self.option_hover['OptionBack'], 
                                        height_current, blit_list, rect_list,
                                        align=1)
        self.option_hover['OptionBack'], height_current = result
        height_current += self.menu_padding * 2
        if self.option_hover['OptionBack'] and mouse_clicked:
            self.update_state('Instructions')
            
        
        self.screen_menu_about = self._new_surface(height_current, blit_list, rect_list)
    
    def draw_surface_menu_instructions(self):
    
        mouse_clicked = self._mouse_click()
        height_current = self.menu_padding
        blit_list = []
        rect_list = []
        
        title_message = 'Instructions'
        subtitle_message = ''
        height_current = self._generate_menu_title(title_message, subtitle_message, height_current, blit_list)
        
        text_chunks = ['The aim of the game is to finish with more points than',
                       'your opponent.',
                       '',
                       'You get one point for completing a row in any direction,',
                       'and the game ends when no more points are available.']
        for message in text_chunks:
            if not message:
                height_current += self.menu_padding
                continue
            result = self._generate_menu_selection(message,
                                            [], [], height_current,
                                            blit_list, rect_list, 
                                            font=self.font_sm_m)
            _, height_current = result
        height_current += self.menu_padding
        
        
        options = ('Previous', 'Next')
        option_len = len(options)
        selected = []
        for i in range(option_len):
            background = False
            foreground = i == self.option_hover['OptionDirectionExample']
            selected.append([background, foreground])
        
        current_grid = self.option_set['OptionDirectionExample'] % self.example_grid_count
        result = self._generate_menu_selection('Here are examples of each direction ({}/{}):'.format(current_grid + 1, self.example_grid_count),
                                        options, selected, height_current,
                                        blit_list, rect_list,
                                        font=self.font_sm_m)
        self.option_hover['OptionDirectionExample'], height_current = result
        if self.option_hover['OptionDirectionExample'] is not None and mouse_clicked:
            self.option_set['OptionDirectionExample'] += self.option_hover['OptionDirectionExample'] * 2 - 1
        
        #Draw grid
        height_current += self.menu_padding
        surface = self.example_grid[current_grid]
        grid_height = height_current
        height_current += surface.get_size()[1] + self.menu_padding
        
        
        
        text_chunks = ["The grid will flip itself to stop things from becoming too",
                       "easy."]
        for message in text_chunks:
            if not message:
                height_current += self.menu_padding
                continue
            result = self._generate_menu_selection(message,
                                            [], [], height_current,
                                            blit_list, rect_list, 
                                            font=self.font_sm_m)
            _, height_current = result
            
            
        height_current += self.menu_padding * 2
        result = self._generate_menu_button('Back',
                                        self.option_hover['OptionBack'], 
                                        height_current, blit_list, rect_list,
                                        align=0)
        self.option_hover['OptionBack'], _ = result
        if self.option_hover['OptionBack'] and mouse_clicked:
            self.update_state('Menu')
            
        
        result = self._generate_menu_button('About',
                                        self.option_hover['OptionAbout'], 
                                        height_current, blit_list, rect_list,
                                        align=2)
        self.option_hover['OptionAbout'], height_current = result
        height_current += self.menu_padding * 2
        if self.option_hover['OptionAbout'] and mouse_clicked:
            self.update_state('About')
            
        
        self.screen_menu_instructions = self._new_surface(height_current, blit_list, rect_list)
        self.screen_menu_instructions.blit(surface, (self.menu_width / 4 + self.scroll_width, grid_height))
        
    def draw_surface_menu_settings(self):
        height_current = self.menu_padding
        blit_list = []
        rect_list = []
        
        #Render menu title
        if self.temp_data['Winner'] is None:
            title_message = 'Connect 3D'
            subtitle_message = 'By Peter Hunt'
        else:
            subtitle_message = 'Want to play again?'
            if len(self.temp_data['Winner']) == 1:
                title_message = 'Player {} was the winner!'.format(self.temp_data['Winner'][0])
            else:
                title_message = 'It was a draw!'
        height_current = self._generate_menu_title(title_message, subtitle_message, height_current, blit_list)
        
        
        instant_restart = all(not i for i in self.game.core.grid) and self.temp_data['PendingMove'] is None
        show_advanced = bool(self.option_set['AdvancedOptions'])
        mouse_clicked = self._mouse_click()
        
        #Player settings
        if True:
        
            #Add or remove players
            if show_advanced:
                temp_height = height_current - self.menu_font_size
                options = ('Add', 'Remove')
                option_len = len(options)
                selected = []
                
                player_count = len(self.option_set['Players'])
                too_high = player_count > len(self.colour_order) - 1
                too_low = player_count < 3
                for i in range(option_len):
                    background = False
                    foreground = i == self.option_hover['PlayerChange']
                    if too_high and not i or too_low and i:
                        foreground = False
                        
                    selected.append([background, foreground])
                    
                result = self._generate_menu_selection('',
                                                options, selected, temp_height,
                                                blit_list, rect_list, centre=True)
                self.option_hover['PlayerChange'], temp_height = result
                
                if self.option_hover['PlayerChange'] is not None and mouse_clicked:
                    changed_players = False
                    
                    if self.option_hover['PlayerChange'] and not too_low:
                        del self.option_set['Players'][-1]
                        del self.option_hover['Players'][-1]
                        changed_players = True
                        print True
                        
                    elif not self.option_hover['PlayerChange'] and not too_high:
                        self.option_set['Players'].append(self.game.ai.DEFAULT_DIFFICULTY + 1)
                        self.option_hover['Players'].append(None)
                        changed_players = True
                        print False
                    
                    if changed_players:
                        self.option_set['ShuffleTurns'] += 1 if self.option_set['ShuffleTurns'] % 2 else -1
                        if instant_restart:
                            self.frame_data['Reload'] = True
                    
            
            #Configure players
            options = ('Human', 'Beginner', 'Easy', 'Medium', 'Hard', 'Extreme')
            option_len = len(options)
            
            for id, player in enumerate(self.option_set['Players']):
            
                selected = []
                dict_name = 'Player{}'.format(id + 1)
                
                for i in range(option_len):
                    if id >= self.game._player_count:
                        background = i == self.game.ai.DEFAULT_DIFFICULTY + 1
                    else:
                        background = i == self.game.players[id]
                    foreground = i in (self.option_set['Players'][id], self.option_hover['Players'][id])
                    selected.append((background, foreground))
                
                result = self._generate_menu_selection('Player {}: '.format(id + 1),
                                                options, selected, height_current,
                                                blit_list, rect_list)
                self.option_hover['Players'][id], height_current = result
                height_current += self.menu_padding
                
                if self.option_hover['Players'][id] is not None and mouse_clicked:
                    self.option_set['Players'][id] = self.option_hover['Players'][id]
                    if instant_restart:
                        self.frame_data['Reload'] = True
        
        
        height_current += self.menu_padding * 2
        
        #End early option
        if show_advanced:
            options = ('Yes', 'No')
            option_len = len(options)
            selected = []
            
            for i in range(option_len):
                background = i == (not self.option_set['EndEarly'])
                foreground = i in (not self.option_set['EndEarly'], self.option_hover['EndEarly'])
                selected.append([background, foreground])
                
            result = self._generate_menu_selection('End when no rows are left?',
                                            options, selected, height_current,
                                            blit_list, rect_list)
            self.option_hover['EndEarly'], height_current = result
            
            if self.option_hover['EndEarly'] is not None and mouse_clicked:
                self.option_set['EndEarly'] = not self.option_hover['EndEarly']
            height_current += self.menu_padding
         
        #Shuffle options
        if True:
            options = ('Mirror/Rotate', 'Mirror', 'No')
            option_len = len(options)
            selected = []
            
            for i in range(option_len):
                background = i == 2 - self.game.core.shuffle_level
                foreground = i in (2 - self.option_set['ShuffleLevel'], self.option_hover['ShuffleLevel'])
                selected.append((background, foreground))
            
            turns = self.option_set['ShuffleTurns']
            result = self._generate_menu_selection('Flip grid every {} turn{}?'.format(turns, '' if turns == 1 else 's'),
                                            options, selected, height_current,
                                            blit_list, rect_list)
            self.option_hover['ShuffleLevel'], height_current = result
            
            if self.option_hover['ShuffleLevel'] is not None and mouse_clicked:
                self.option_set['ShuffleLevel'] = 2 - self.option_hover['ShuffleLevel']
                if instant_restart:
                    self.frame_data['Reload'] = True
            height_current += self.menu_padding
            
            
            
            #Increase or decrease shuffle count
            options = ('Increase', 'Decrease')
            option_len = len(options)
            selected = []
            
            shuffle_turns_max = max(1, self.game.core._size_cubed // 2)
            too_low = self.option_set['ShuffleTurns'] <= 2
            too_high = self.option_set['ShuffleTurns'] > shuffle_turns_max
            if show_advanced:
            
                for i in range(option_len):
                    background = False
                    foreground = i == self.option_hover['ShuffleTurns']
                    if too_low and i:
                        foreground = False
                    if too_high and not i:
                        foreground = False
                    selected.append((background, foreground))
                    
                result = self._generate_menu_selection('',
                                                options, selected, height_current,
                                                blit_list, rect_list, centre=True)
                self.option_hover['ShuffleTurns'], height_current = result
                
                if self.option_hover['ShuffleTurns'] is not None and mouse_clicked:
                    if self.option_hover['ShuffleTurns'] == 1 and not too_low:
                        self.option_set['ShuffleTurns'] -= 2
                    elif self.option_hover['ShuffleTurns'] == 0 and not too_high:
                        self.option_set['ShuffleTurns'] += 2
                        
                    if instant_restart:
                        self.frame_data['Reload'] = True
                height_current += self.menu_padding
        
            while self.option_set['ShuffleTurns'] > shuffle_turns_max:
                self.option_set['ShuffleTurns'] -= 2
        
        height_current += self.menu_padding * 2
                  

        #Time limit options
        if True:
            options = ('Yes', 'No')
            option_len = len(options)
            selected = []
            
            for i in range(option_len):
                background = i == (not self.timer_enabled)
                foreground = i in (not self.timer_enabled, self.option_hover['TimeEnabled'])
                selected.append([background, foreground])
                
            result = self._generate_menu_selection('Use a turn time limit?',
                                            options, selected, height_current,
                                            blit_list, rect_list)
            self.option_hover['TimeEnabled'], height_current = result
            
            if self.option_hover['TimeEnabled'] is not None and mouse_clicked:
                self.timer_enabled = not self.option_hover['TimeEnabled']
            height_current += self.menu_padding
                
                
            #Ask about time options
            if self.timer_enabled and show_advanced:
                options = ('Increase', 'Decrease')
                option_len = len(options)
                selected = []
                
                too_low = self.timer_count <= 10
                for i in range(option_len):
                    background = False
                    foreground = i == self.option_hover['TimeChange']
                    if too_low:
                        foreground = False
                    selected.append([background, foreground])
                    
                timer = self.TIMER_DEFAULT if not self.timer_count else self.timer_count
                result = self._generate_menu_selection('Limited to {} seconds.'.format(timer),
                                                options, selected, height_current,
                                                blit_list, rect_list)
                self.option_hover['TimeChange'], height_current = result
                
                #If clicked and held, only run once to stop an increment every frame
                if self.option_hover['TimeChange'] is not None and mouse_clicked and not too_low:
                    self.timer_count += 10 * (-1 if self.option_hover['TimeChange'] else 1)
                    if self.temp_data['MoveTimeLeft']:
                        self.temp_data['MoveTimeLeft'] = self.TICKS * self.timer_count
            
        
        height_current += self.menu_padding * 2
        
            
        #Grid size options
        if show_advanced:
        
            height_current += self.menu_padding
            options = ('Increase', 'Decrease')
            option_len = len(options)
            selected = []
            
            too_low = self.option_set['GridSize'] == 1
            for i in range(option_len):
                background = False
                foreground = i == self.option_hover['GridSize']
                if too_low and i:
                    foreground = False
                selected.append((background, foreground))
            
            result = self._generate_menu_selection('Grid size is {}.'.format(self.option_set['GridSize']),
                                            options, selected, height_current,
                                            blit_list, rect_list)
            self.option_hover['GridSize'], height_current = result
            
            if (self.option_hover['GridSize'] is not None and mouse_clicked 
                and not(too_low and self.option_hover['GridSize'])):
                
                self.option_set['GridSize'] += 1 - self.option_hover['GridSize'] * 2
                if instant_restart:
                    self.frame_data['Reload'] = True
            height_current += self.menu_padding
        
            #Slow warning
            if self.option_set['GridSize'] > 5 and any(self.option_set['Players']):
                result = self._generate_menu_selection('Warning: The AI will run extremely slow!',
                                                [], [], height_current,
                                                blit_list, rect_list, centre=True)
                _, height_current = result
    
                height_current += self.menu_padding
            height_current += self.menu_padding
        
        
            #Debug option
            try:
                d_pressed = self.frame_data['Keys'][pygame.K_d]
            except KeyError:
                pass
            else:
                if d_pressed and self.frame_data['KeyShift'] and self.frame_data['KeyAlt']:
                
                    options = ('Yes', 'No')
                    option_len = len(options)
                    selected = []
                    
                    for i in range(option_len):
                        background = i == (not self.option_set['Debug'])
                        foreground = i in (not self.option_set['Debug'], self.option_hover['Debug'])
                        selected.append([background, foreground])
                        
                    result = self._generate_menu_selection('Show debug information?',
                                                    options, selected, height_current,
                                                    blit_list, rect_list)
                    self.option_hover['Debug'], height_current = result
                    
                    if self.option_hover['Debug'] is not None and mouse_clicked:
                        self.option_set['Debug'] = not self.option_hover['Debug']
                    height_current += self.menu_padding * 2
            
                
        
        #Menu buttons
        if instant_restart or self.temp_data['Winner'] is not None:
            result = self._generate_menu_button('Start Game',
                                            self.option_hover['OptionNewGame'], 
                                            height_current, blit_list, rect_list,
                                            align=2)
            self.option_hover['OptionNewGame'], _ = result
            if self.option_hover['OptionNewGame'] and mouse_clicked:
                self.frame_data['Reload'] = True
                self.update_state('Main')
                
            
            result = self._generate_menu_button('Instructions',
                                            self.option_hover['OptionInstructions'], 
                                            height_current, blit_list, rect_list,
                                            align=0)
            self.option_hover['OptionInstructions'], height_current = result
            height_current += self.menu_padding * 2
            if self.option_hover['OptionInstructions'] and mouse_clicked:
                self.update_state('Instructions')
            
            result = self._generate_menu_button('Quit To Desktop',
                                            self.option_hover['OptionQuit'], 
                                            height_current, blit_list, rect_list,
                                            align=1)
            self.option_hover['OptionQuit'], height_current = result
            height_current += self.menu_padding
            if self.option_hover['OptionQuit'] and mouse_clicked:
                self.exit()
        
        else:
            result = self._generate_menu_button('Instructions',
                                            self.option_hover['OptionInstructions'], 
                                            height_current, blit_list, rect_list,
                                            align=1)
            self.option_hover['OptionInstructions'], height_current = result
            height_current += self.menu_padding * 2
            if self.option_hover['OptionInstructions'] and mouse_clicked:
                self.update_state('Instructions')
                
            result = self._generate_menu_selection('Restart game to apply settings.',
                                            [], [], height_current,
                                            blit_list, rect_list, centre=True)
            _, height_current = result
            height_current += self.menu_padding * 2
            
            result = self._generate_menu_button('Continue',
                                            self.option_hover['OptionContinue'], 
                                            height_current, blit_list, rect_list,
                                            align=0)
            self.option_hover['OptionContinue'], _ = result
            if self.option_hover['OptionContinue'] and mouse_clicked:
                self.update_state('Main')
            
            result = self._generate_menu_button('New Game',
                                            self.option_hover['OptionNewGame'], 
                                            height_current, blit_list, rect_list,
                                            align=2)
            self.option_hover['OptionNewGame'], height_current = result
            height_current += self.menu_padding * 2
            if self.option_hover['OptionNewGame'] and mouse_clicked:
                self.frame_data['Reload'] = True
                self.update_state('Main')
                
                
            result = self._generate_menu_button('Quit To Desktop',
                                            self.option_hover['OptionQuit'], 
                                            height_current, blit_list, rect_list,
                                            align=1)
            self.option_hover['OptionQuit'], height_current = result
            height_current += self.menu_padding
            if self.option_hover['OptionQuit'] and mouse_clicked:
                self.exit()
            
        #Ask about advanced options
        if True:
            height_current += self.menu_padding
            options = ('Yes', 'No')
            option_len = len(options)
            selected = []
            
            for i in range(option_len):
                background = i == (not show_advanced)
                foreground = background or i == self.option_hover['AdvancedOptions']
                selected.append([background, foreground])
                
            result = self._generate_menu_selection('Show advanced options?',
                                            options, selected, height_current,
                                            blit_list, rect_list, centre=True,
                                            important=False)
            self.option_hover['AdvancedOptions'], height_current = result
            height_current += self.menu_padding
            
            if self.option_hover['AdvancedOptions'] is not None and mouse_clicked:
                self.option_set['AdvancedOptions'] = not self.option_hover['AdvancedOptions']
        
        self.screen_menu_background = self._new_surface(height_current, blit_list, rect_list)
    
    def draw_surface_debug(self):
        
        if not self.option_set['Debug']:
            pygame.display.set_caption('Connect 3D')
            return
            
        if self.frame_data['GameTime'].fps:
            pygame.display.set_caption('Framerate: {}'.format(self.frame_data['GameTime'].fps))
        
        try:
            self.frame_data['MousePos']
            self.last_fps
        except KeyError:
            return
        except AttributeError:
            self.last_fps = self.frame_data['GameTime'].fps 
    
        self.surface_debug = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA, 32)
        
        #Draw main debug info to the bottom left
        if self.frame_data['GameTime'].fps is not None:
            self.last_fps = self.frame_data['GameTime'].fps 
            
        info = ['GRID:',
                ' Size: {}'.format(self.game.core.size),
                ' Length: {}'.format(int(self.draw.length)),
                ' Angle: {}'.format(self.draw.angle),
                ' Offset: {}'.format(map(int, self.draw.offset)),
                'MOUSE:',
                ' Coordinates: {}'.format(self.frame_data['MousePos']),
                ' Block ID: {}'.format(self.draw.game_to_block_index(*self.frame_data['MousePos'])),
                'PYGAME:',
                ' Redraw: {}'.format(self.frame_data['Redraw']),
                ' Ticks: {}'.format(self.frame_data['GameTime'].total_ticks),
                ' FPS: {}'.format(self.last_fps),
                ' Desired FPS: {}'.format(self.frame_data['GameTime']._temp_fps or self.frame_data['GameTime'].GTObject.desired_fps)
                ]
                
        fonts = [self.font_sm.render(format_text(i), 1, BLACK) for i in info]
        sizes = [i.get_rect()[2:] for i in fonts]
        for i in range(len(info)):
            message_height = self.HEIGHT - sum(j[1] for j in sizes[i:])
            self.surface_debug.blit(fonts[i], (self.text_padding, message_height))
        
        
        #Draw AI debug info to the bottom right
        if self.game._ai_text:
            
            fonts = [self.font_sm.render(format_text(i), 1, BLACK) for i in self.game._ai_text]
            sizes = [i.get_rect()[2:] for i in fonts]
            for i in range(len(self.game._ai_text)):
                message_height = self.HEIGHT - sum(j[1] for j in sizes[i:])
                self.surface_debug.blit(fonts[i], (self.WIDTH - sizes[i][0] - self.text_padding, message_height))
        
        
        #Redraw the background
        self.screen.blit(self.background, (0, 0))
        if self.state != 'Main':
            location = list(self.menu_location)
            location[0] += self.scroll_width // 2
            self.screen.blit(self.screen_menu_holder, location)
        self.screen.blit(self.surface_debug, (0, 0))
        
        self.frame_data['Redraw'] = True
  
    def _generate_menu_title(self, title, subtitle, height_current, blit_list):
        
        font = self.font_lg_m.render(title, 1, BLACK)
        blit_list.append((font, (self.width_padding * 2, height_current)))
        height_current += font.get_rect()[3] + self.menu_padding
        
        if subtitle:
            font = self.font_md_m.render(subtitle, 1, BLACK)
            blit_list.append((font, (self.width_padding * 2, height_current)))
            height_current += self.menu_padding * 6
        
        return height_current
    
    def _generate_menu_button(self, message, hover, height_current, 
                              blit_list, rect_list, align=1):
        multiplier = 1
        
        #Set up text
        colour = BLACK if hover else GREY
        font = self.font_lg_m.render(message, 1, colour)
        size = font.get_rect()[2:]
        
        offset = ((self.menu_width - self.scroll_width) * (0.5 + align / 2) - size[0]) / 2
        
        square = (offset - self.menu_box_padding,
                  height_current - self.menu_box_padding / 2,
                  size[0] + self.menu_box_padding * 2,
                  size[1] + self.menu_box_padding)
    
        #rect_list.append([self.menu_colour, square])
        blit_list.append((font, (offset, height_current)))
        height_current += square[3]
        
        hovering = False
        
        #Detect if mouse is over it
        try:
            x, y = self.frame_data['MousePos']
        except (AttributeError, KeyError):
            pass
        else:
            x -= self.menu_location[0] + self.scroll_width // 2
            y -= self.menu_location[1] + self.scroll_offset
            x_selected = square[0] < x < square[0] + square[2]
            y_selected = square[1] < y < square[1] + square[3]
            if x_selected and y_selected:
                hovering = True
            
        return (hovering, height_current)
        
    def _generate_menu_selection(self, message, options, selected, 
                                 height_current, blit_list, rect_list, 
                                 centre=False, font=None, important=True):
        
        font_type = font or self.font_md_m
        
        font = font_type.render('{} '.format(message), 1, BLACK if important else SELECTION['Default'][1])
        start_size = font.get_rect()[2:]
        fonts = [font_type.render(option, 1, BLACK) for option in options]
        sizes = [option.get_rect()[2:] for option in fonts]
        
        offset = 0
        if centre:
            size_sum = sum(i[0] + 2 for i in sizes) + start_size[0]
            offset = (self.menu_width - size_sum - self.scroll_width / 2) // 2
    
        if message:
            start_size[0] += 2
            blit_list.append((font, (self.width_padding * 2 + offset, height_current)))
        
        
        
        #Calculate square sizes
        square_list = []
        num_options = len(options)
        for i, size in enumerate(sizes):
            width_offset = (sum(j[0] + 2 for j in sizes[:i])
                            + self.width_padding * (i + 1) #gap between the start
                            + start_size[0] + offset)
            
            #Correctly apply width offset
            if not i:
                width_offset1 = width_offset - self.menu_box_padding / 2
                width_offset2 = size[0] + self.menu_box_padding
            elif i == num_options - 1:
                width_offset1 = width_offset - self.menu_box_padding / 4
                width_offset2 = size[0] + self.menu_box_padding / 2
            else:
                width_offset1 = width_offset - self.menu_box_padding / 4
                width_offset2 = size[0] + self.menu_box_padding / 2
                
            square = (width_offset1,
                     height_current - self.menu_box_padding / 4,
                     width_offset2,
                     size[1] + self.menu_box_padding / 4)
            square_list.append(square)
    
            #Set colours
            order = ('Background', 'Foreground')
            
            colours = list(SELECTION['Default'])
            for j, selection in enumerate(selected[i]):
                if selection:
                    rect_colour, text_colour = list(SELECTION[order[j]])
                    if rect_colour is not None:
                        colours[0] = rect_colour
                    if text_colour is not None:
                        colours[1] = text_colour
            rect_colour, text_colour = colours
            if rect_colour == True:
                rect_colour = self.menu_colour
            if text_colour == True:
                text_colour = self.menu_colour
            
            #Add to list
            font = font_type.render(options[i], 1, text_colour)
            blit_list.append((font, (width_offset, height_current)))
            rect_list.append([rect_colour, square])
    
        selected_block = None
        if options:
            try:
                x, y = self.frame_data['MousePos']
            except (AttributeError, KeyError):
                pass
            else:
                x -= self.menu_location[0] + self.scroll_width // 2
                y -= self.menu_location[1] + self.scroll_offset
                for i, square in enumerate(square_list):
                    x_selected = square[0] < x < square[0] + square[2]
                    y_selected = square[1] < y < square[1] + square[3]
                    if x_selected and y_selected:
                        selected_block = i
        
        height_current += start_size[1]
        return (selected_block, height_current)
    
    def _run_ai(self, run=True):
        """Runs the AI in a thread.
        Until _ai_move or _ai_state is not None, it is not completed.
        """
        self.game._ai_move = self.game._ai_state = None
        if run:
            ThreadHelper(self.game.ai.calculate_move, 
                         self.game._player, 
                         difficulty=self.game._player_types[self.game._player - 1], 
                         _range=self.game._range_players).start()
    
    def reload_game(self):
        old_size = self.game.core.size
        
        try:
            self.game = Connect3DGame(players=self.option_set['Players'], 
                                      shuffle_level=self.option_set['ShuffleLevel'],
                                      shuffle_turns=self.option_set['ShuffleTurns'],
                                      size=self.option_set['GridSize'])
        except AttributeError:
            pass
            
        else:
        
            #Recalculate grid coordinates
            if old_size != self.game.core.size:
                self.draw = DrawData(self.game.core, self.draw.length, self.draw.angle, self.draw.padding, self.draw.offset)
                self.resize_screen()
            
            if self.game._player == -1:
                self.game._player = random.choice(self.game._range_players)
            
            #Update difficulty
            try:
                for i in range(self.game._player_count):
                    self.game.players[i] = bytearray(self.option_set['Player{}'.format(i + 1)])
            except KeyError:
                pass
                
        #Reset any game data
        self.temp_data = {'Hover': None,
                          'PendingMove': None,
                          'EarlyHover': None,
                          'Winner': None,
                          'ShuffleCount': 0,
                          'Flipped': False,
                          'Skipped': False,
                          'MoveTimeLeft': None}
        self.game_ended = False
        
        
        #Save things that should persist after reset
        try:
            advanced_options = self.option_set['AdvancedOptions']
            scroll_position = self.option_set['Scroll']
            end_early = self.option_set['EndEarly']
            grid_hover = self.option_hover['GridSize']
            debug = self.option_set['Debug']
        except AttributeError:
            advanced_options = None
            scroll_position = 0
            end_early = True
            grid_hover = 0
            debug = False
        
        self.option_set = {'Debug': None,
                           'Scroll': None,
                           'ShuffleLevel': None,
                           'TimeEnabled': None,
                           'TimeChange': None,
                           'PlayerChange': None,
                           'AdvancedOptions': None,
                           'Players': [None for _ in self.game._range_players],
                           'GridSize': None,
                           'ShuffleTurns': None,
                           'EndEarly': None,
                           'OptionNewGame': None,
                           'OptionInstructions': None,
                           'OptionAbout': None,
                           'OptionBack': None,
                           'OptionQuit': None,
                           'OptionContinue': None,
                           'OptionDirectionExample': None}
        self.option_hover = dict(self.option_set)
        
        self.option_hover['GridSize'] = grid_hover
        self.option_set['Debug'] = debug
        self.option_set['OptionDirectionExample'] = 0
        self.option_set['EndEarly'] = end_early
        self.option_set['AdvancedOptions'] = advanced_options
        self.option_set['Scroll'] = scroll_position
        self.option_set['ShuffleLevel'] = self.game.core.shuffle_level
        self.option_set['TimeEnabled'] = not self.timer_count
        self.option_set['Players'] = list(self.game.players)
        self.option_set['GridSize'] = self.game.core.size
        self.option_set['ShuffleTurns'] = self.game.shuffle_turns
        self.scroll_offset = 0
    
    def play(self):
    
        #Initialise screen
        pygame.init()
        self.frame_data = {'MouseClick': list(pygame.mouse.get_pressed())}
        self.flag_data = {'Disable': [None, None, None],
                          'LastState': None}
        
        #Import the font
        self.font_file = 'Miss Monkey.ttf'
        try:
            pygame.font.Font(self.font_file, 0)
        except IOError:
            raise IOError('failed to load font')
        
        #Adjust width and height to fit on screen
        self.height_ratio = 0.64
        self.HEIGHT = int(pygame.display.Info().current_h * 0.88 / 16) * 16
        self.WIDTH = int(self.HEIGHT * self.height_ratio)
        self.update_state('Menu')
        
        self.reload_game()
        self.resize_screen()
        
        '''
        #Clipboard
        pygame.scrap.init()
        try:
            pygame.scrap.put(pygame.SCRAP_TEXT, self.game.__repr__())
        except pygame.error:
            pass
            '''
        
        GT = GameTime(self.FPS_MAIN, self.TICKS)
        tick_count = 0
        while True:
            with GameTimeLoop(GT) as game_time:
            
                #Store frame specific things so you don't need to call it multiple times
                self.frame_data = {'GameTime': game_time,
                                   'Redraw': False,
                                   'Events': pygame.event.get(),
                                   'Keys': pygame.key.get_pressed(),
                                   'MousePos': pygame.mouse.get_pos(),
                                   'MouseClick': list(pygame.mouse.get_pressed()) + [0, 0],
                                   'Reload': False}
                
                self.frame_data['MouseUse'] = any(self.frame_data['MouseClick']) or any(self.frame_data['Keys'])
                self.frame_data['KeyCTRL'] = self.frame_data['Keys'][pygame.K_RCTRL] or self.frame_data['Keys'][pygame.K_LCTRL]
                self.frame_data['KeyAlt'] = self.frame_data['Keys'][pygame.K_RALT] or self.frame_data['Keys'][pygame.K_LALT]
                self.frame_data['KeyShift'] = self.frame_data['Keys'][pygame.K_RSHIFT] or self.frame_data['Keys'][pygame.K_LSHIFT]
                    
                #Handle quitting and resizing window
                if self.state is None:
                    pygame.quit()
                    return
                
                for event in self.frame_data['Events']:
                    if event.type == pygame.QUIT:
                        self.update_state(None)
                        
                    elif event.type == pygame.VIDEORESIZE:
                        self.WIDTH, self.HEIGHT = event.dict['size']
                        self.resize_screen()
                        self.frame_data['Redraw'] = True
                        
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        self.frame_data['MouseUse'] = True
                        self.frame_data['MouseClick'][event.button - 1] = 1
                        
                    elif event.type == pygame.MOUSEBUTTONUP:
                        self.frame_data['MouseUse'] = True
                    
                    elif event.type == pygame.MOUSEMOTION:
                        self.frame_data['MouseUse'] = True
                        
                    elif event.type == pygame.KEYDOWN:
                        self.frame_data['MouseUse'] = True
                        if event.key == pygame.K_ESCAPE:
                            if self.state == 'Main':
                                self.update_state('Menu')
                            else:
                                self.update_state('Main')
                                
                    elif event.type == pygame.KEYUP:
                        self.frame_data['MouseUse'] = True
                
                    
                if self.frame_data['MouseUse']:
                    self.frame_data['Redraw'] = True
                    
                #---MAIN LOOP START---#
                
                if self.state == 'Main':
                    self.loop_main()
                
                elif self.state in ('Menu', 'Instructions', 'About'):
                    self.loop_menu()
                
                self.draw_surface_debug()
                
                
                #---MAIN LOOP END---#
                if self.frame_data['Redraw']:
                    pygame.display.flip()
                    
                if self.frame_data['Reload']:
                    self.reload_game()

    def _mouse_click(self, i=0):
        """Disable more than one click happening when holding the mouse button down."""
        mouse_clicked = self.frame_data['MouseClick'][i] and self.flag_data['Disable'][i] is None
        if self.flag_data['Disable'][i] and not self.frame_data['MouseClick'][i]:
            self.flag_data['Disable'][i] = None
        elif self.frame_data['MouseClick'][i] and self.flag_data['Disable'][i] is None:
            self.flag_data['Disable'][i] = True
        return mouse_clicked
    
    def loop_main(self):
    
        #End the current game
        if self.temp_data['Winner'] is not None:
            self.draw_surface_main_title()
            self.update_state('Menu')
            
            #Only submit info once
            if not self.game_ended:
                self.game_ended = True
                
                ifttt_key = 'bnHbA0p4ecI5X0eaoxm7CdzEKIek_9hI1u7fUfIl8HR'
                ifttt_name = 'Connect3D'
                ifttt_url = 'https://maker.ifttt.com/trigger/{}/with/key/{}'
                headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
                ifttt_content = [
                    'Version: {}'.format(VERSION),
                    'Date: {}'.format(time.time()),
                    'Time taken: {}'.format(self.frame_data['GameTime'].total_ticks / self.TICKS),
                    'Winner: {}'.format(self.temp_data['Winner']),
                    'Players: {}'.format(list(self.game.players)),
                    #'Grid Data: {}'.format(','.join(map(str, list(self.game.core.grid)))),
                    'Grid Data: {}'.format(''.join('{0:08b}'.format(i) for i in self.game.core.grid)),
                    'Size: {}'.format(self.game.core.size),
                    'Shuffle Level: {}'.format(self.game.core.shuffle_level),
                    'Shuffle Turns: {}'.format(self.game.shuffle_turns),
                    'Turn Time: {}'.format(self.timer_count if self.timer_enabled else 0),
                    'Debug: {}'.format(self.option_set['Debug'])
                ]
                try:
                    req = urllib2.Request(ifttt_url.format(ifttt_name, ifttt_key))
                    req.add_header('Content-Type', 'application/json')
                    response = urllib2.urlopen(req, json.dumps({'value1': '<br>'.join(ifttt_content)}))
                except urllib2.URLError:
                    pass
            
            return
    
        mouse_click_l = self._mouse_click(0)
        mouse_click_r = self._mouse_click(2)
    
        #Count ticks down
        force_end = False
        if self.temp_data['MoveTimeLeft'] is not None:
            old_time_left = self.temp_data['MoveTimeLeft']
            self.temp_data['MoveTimeLeft'] -= self.frame_data['GameTime'].ticks
            if old_time_left // self.TICKS != self.temp_data['MoveTimeLeft'] // self.TICKS:
                
                self.frame_data['Redraw'] = True
                self.draw_surface_main_time()
            
            #Skip go if time ran out
            if self.temp_data['MoveTimeLeft'] < 0:
                force_end = True + (self.temp_data['PendingMove'] is not None)
                self.draw_surface_main_title()
        
        player_type = self.game._player_types[self.game._player - 1]
        
        #Moved mouse
        mouse_block_id = None
        if self.frame_data['MouseUse']:
            
            self.frame_data['GameTime'].temp_fps(self.FPS_MAIN)
            mouse_block_id = self.draw.game_to_block_index(*self.frame_data['MousePos'])
            
            #Disable mouse if winner
            if self.temp_data['Winner'] is not None:
                mouse_block_id = None
            
            #Enemy has finished their go, gets rid of the 'frozen game' effect
            if self.temp_data['PendingMove'] is not None and self.temp_data['PendingMove'][3]:
                next_player = self.game.next_player(self.game._player, self.game._player_count)
                
                #Re-activate hover if next player is human
                if self.game._player_types[next_player - 1] < 0:
                    self.temp_data['Hover'] = [mouse_block_id, next_player]
                    
                else:
                    self.temp_data['Hover'] = None
            
            #Players go
            elif player_type < 0:
                if self.temp_data['PendingMove'] is None:
                    self.temp_data['Hover'] = [mouse_block_id, self.game._player]
                else:
                    self.temp_data['Hover'] = None
                    
            else:
                self.temp_data['Hover'] = None
            
            self.draw_surface_main_title()
            self.draw_surface_main_grid()
            
        
        #Move not yet made
        if (self.temp_data['PendingMove'] is None or not self.temp_data['PendingMove'][3]) and not force_end:
        
            #Human player
            if player_type < 0:
            
                if self.temp_data['MoveTimeLeft'] is None and self.timer_count:
                   self.temp_data['MoveTimeLeft'] = self.TICKS * self.timer_count
                   self.draw_surface_main_time()
                    
                #Mouse button clicked
                if self.frame_data['MouseClick'][0]:
                    
                    #Player has just clicked
                    if self.temp_data['PendingMove'] is None:
                        if mouse_block_id is not None and not self.game.core.grid[mouse_block_id] and mouse_click_l:
                            self.temp_data['PendingMove'] = [mouse_block_id, 
                                                             self.frame_data['GameTime'].total_ticks + self.MOVE_WAIT, 
                                                             True, 
                                                             False]
                    
                    #Player is holding click over the block
                    elif mouse_block_id == self.temp_data['PendingMove'][0]:
                        self.temp_data['PendingMove'][2] = True
                    
                    #Player is holding click and has moved mouse away
                    elif mouse_block_id != self.temp_data['PendingMove'][0]:
                        self.temp_data['PendingMove'][2] = False
                        
                    #Cancel the click
                    if mouse_click_r:
                        self.temp_data['PendingMove'] = None
                
                    self.draw_surface_main_grid()
                    self.frame_data['Redraw'] = True
                
                #Mouse button released
                elif self.temp_data['PendingMove'] is not None and not self.temp_data['PendingMove'][3]:
                    
                    #If mouse was kept on
                    if self.temp_data['PendingMove'][2]:
                        self.temp_data['PendingMove'][3] = True
                    
                    #If mouse was dragged off
                    else:
                        self.temp_data['PendingMove'] = None
                    
                    self.draw_surface_main_grid()
                    self.frame_data['Redraw'] = True
                
            
            #Computer player
            elif self.game._ai_running is False:
            
                if self.temp_data['Winner'] is None:
                
                    #Move has not started yet
                    if self.game._ai_move is None:
                        self._run_ai()
                    
                    #Move finished calculating
                    else:
                        self.temp_data['PendingMove'] = [self.game._ai_move, 
                                                         self.frame_data['GameTime'].total_ticks + self.MOVE_WAIT, 
                                                         True, 
                                                         True]
                                                         
                        self._run_ai(run=False)
                        
                self.draw_surface_main_title()
                self.draw_surface_main_grid()
                self.frame_data['Redraw'] = True
            else:
                self.draw_surface_main_title()
                self.frame_data['Redraw'] = True
        
        #Commit the move
        else:
            
            #Moved cancelled
            if mouse_click_r and player_type < 0:
                self.temp_data['PendingMove'] = None
                
            elif self.temp_data['PendingMove'] is not None:
                block_id, wait_until, hovering, accept = self.temp_data['PendingMove']
                
                #Cancelled move
                if not accept and not hovering and not mouse_click_l:
                    self.temp_data['PendingMove'] = block_id = wait_until = None
                    
                if (block_id is not None and accept and self.frame_data['GameTime'].total_ticks > wait_until
                    or force_end):
                    self.temp_data['MoveTimeLeft'] = None
                    self.temp_data['PendingMove'] = None
                    self.temp_data['Skipped'] = False
                    self.game.core.grid[block_id] = self.game._player
                    self.game._player = self.game.next_player(self.game._player, self.game._player_count)
                
                    #Shuffle grid
                    self.temp_data['ShuffleCount'] += 1
                    if self.temp_data['ShuffleCount'] >= self.game.shuffle_turns:
                        self.temp_data['ShuffleCount'] = 0
                        self.temp_data['Flipped'] = self.game.core.shuffle()
                    else:
                        self.temp_data['Flipped'] = False
                    
                    self.draw_surface_main_grid()
                    self.frame_data['Redraw'] = True
                    self.game.core.calculate_score()
        
            self.draw_surface_main_title()
            self.frame_data['Redraw'] = True
            self.temp_data['Winner'] = self.game.check_game_end(self.option_set['EndEarly'])
        
        
        if force_end:
            if force_end == 1:
                self.game._player = self.game.next_player(self.game._player, self.game._player_count)
            self.temp_data['MoveTimeLeft'] = None
            self.temp_data['Skipped'] = force_end
            self.draw_surface_main_title()
                
        #Draw frame
        elif self.frame_data['Redraw']:
            #print self.frame_data['GameTime'].ticks
            self.draw_surface_main_background()
            self.screen.blit(self.background, (0, 0))

    def loop_menu(self):

        #Fix to force it to show menu after someone wins
        if self.flag_data['LastState'] == 'Main':
            self.update_state(update_state=False)
            self.flag_data['LastState'] = None
            self.flag_data['Redraw'] = True
            
        if self.frame_data['Redraw']:
        
            #Run the initial calculations
            mouse_wheel_scroll = self.draw_surface_menu_container()
            if self.state == 'Menu':
                self.draw_surface_menu_settings()
            elif self.state == 'Instructions':
                self.draw_surface_menu_instructions()
            elif self.state == 'About':
                self.draw_surface_menu_about()
            
            
            #Rerun calculations to update the hovering
            if self.state == 'Menu':
                self.draw_surface_menu_settings()
            elif self.state == 'Instructions':
                self.draw_surface_menu_instructions()
            elif self.state == 'About':
                self.draw_surface_menu_about()
            self.draw_surface_menu_container(update_scroll=False)
            
            #Debugging redraws the menu, so don't redraw here
            if not self.option_set['Debug']:
                self.screen.blit(self.background, (0, 0))
            
                location = list(self.menu_location)
                location[0] += self.scroll_width // 2
                self.screen.blit(self.screen_menu_holder, location)

            
c = Connect3DGame(players=(0, True))
c.play()
