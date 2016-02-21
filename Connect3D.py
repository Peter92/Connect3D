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
try:
    import pygame
except ImportError:
    pygame = None
VERSION = '1.1.0'
    
BACKGROUND = (252, 252, 255)
LIGHTBLUE = (86, 190, 255)
LIGHTGREY = (200, 200, 200)
GREY = (128, 128, 128)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PURPLE = (255, 120, 235)

SELECTION = {'Default': [WHITE, LIGHTGREY],
             'Hover': [None, BLACK],
             'Waiting': [None, BLACK],
             'Selected': [GREEN, None]}
             
             
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
    """This gets called every loop but uses GameTime."""
    
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
        self.GTObject.desired_fps = fps
    
    def temp_fps(self, fps):
        self._temp_fps = fps
    
    def update_ticks(self, ticks):
        self.GTObject.start_time = time.time()
        self.GTObject.desired_ticks = ticks
        self.ticks = 0  
             
             
class ThreadHelper(Thread):
    def __init__(self, function, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        Thread.__init__(self)
        self.function = function

    def run(self):
        self.function(*self.args, **self.kwargs)
        
        
def mix_colour(*args):
    mixed_colour = [0, 0, 0]
    num_colours = len(args)
    for colour in range(3):
        mixed_colour[colour] = sum(i[colour] for i in args) / num_colours
    return mixed_colour
    
    
def get_max_keys(x):
    """Return a list of every key containing the max value.
    
    Parameters:
        x (dict): Dictionary to sort and get highest value.
            It must be a dictionary of integers to work properly.
    """
    if x:
        sorted_dict = sorted(x.iteritems(), key=itemgetter(1), reverse=True)
        if sorted_dict[0][1]:
            return sorted([k for k, v in x.iteritems() if v == sorted_dict[0][1]])
    return []
    
            
def split_list(x, n):
    """Split a list by n characters."""
    n = int(n)
    return [x[i:i+n] for i in range(0, len(x), n)]
    
    
def join_list(x):
    """Convert nested lists into one single list."""
    return [j for i in x for j in i]


class Connect3D(object):
    """Class for holding the game information.
    Supports numbers up to 255.
    """
    DEFAULT_SIZE = 4
    DEFAULT_SHUFFLE_LEVEL = 1
    
    def __init__(self, size=None, shuffle_level=None, player=None, num_players=None):
        
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
        output = base64.b64encode(zlib.compress('{}'.format(self.grid)))
        return "Connect3D({}).load('{}')".format(self.size, output)

    def load(self, data, update_score=True):
        grid = bytearray(zlib.decompress(base64.b64decode(data)))
        return self.set_grid(grid, update_score=update_score)
        
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
        for movement, invalid in self.directions.reverse_directions:
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
    
    def __init__(self, shuffle_level=None, shuffle_turns=None, size=None):
        self.shuffle_turns = self.DEFAULT_SHUFFLE_TURNS if shuffle_turns is None else max(0, shuffle_turns)
        self.core = Connect3D(size=size, shuffle_level=shuffle_level)
        self.ai = ArtificialIntelligence(self)
        self._ai_text = []
        self._ai_move = None
        self._ai_state = None
        self._ai_running = False
        
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
        print self.core.grid
        output = base64.b64encode(zlib.compress('{}'.format(self.core.grid)))
        return "Connect3DGame.load('{}')".format(output)
        
    @classmethod
    def load(cls, data):
        """Load a grid into the game."""
        
        grid = bytearray(zlib.decompress(base64.b64decode(data)))
        cube_root = pow(len(grid), 1/3)
        
        if round(cube_root) != round(cube_root, 4):
            raise ValueError('incorrect input size')
            
        #Create new class
        new_instance = cls(size=int(round(cube_root)))
        new_instance.core.set_grid(grid)
        return new_instance
    
    def check_game_end(self, _range_players, end_early=True):
    
            #Check if any points are left to gain
            points_left = True
            if end_early:
                potential_points = {j: Connect3D(self.core.size).set_grid([j if not i else i for i in self.core.grid]).score for j in _range_players}
                if all(self.core.score == potential_points[player] for player in _range_players):
                    points_left = False
                    
            #Check if no spaces are left
            if 0 not in self.core.grid or not points_left:
                return get_max_keys(self.core.score)
                
    
    def play(self, players=(0, 3), basic=False):
        
        num_players = len(players)
        _range_players = [i + 1 for i in range(num_players)]
        current_player = random.choice(_range_players)
        
        if pygame and not basic:
            return GameCore(self).play(players=players)
        
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
            
            winning_player = self.check_game_end(_range_players)
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
                    return self.play(players=players, basic=basic)
                else:
                    return
            
            
            print "Player {}'s turn".format(current_player)
            player_type = players[current_player - 1] - 1
            
            if player_type != -1:
                new_go = self.ai.calculate_move(current_player, difficulty=player_type, _range=_range_players)
                self.core.grid[new_go] = current_player
                
            else:
                #Get and validate input
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
                        self.core.grid[new_go] = current_player
                        break
                    if new_go > max_go:
                        print 'input must be between 0 and {}'.format(max_go)
                    elif self.core.grid[new_go]:
                        print 'input is taken'
                    else:
                        print 'unknown error with input'
            
            current_player = self.next_player(current_player, num_players)
            
            #Flip the grid
            count_shuffle += 1
            if count_shuffle >= self.shuffle_turns:
                count_shuffle = 0
                self.core.shuffle()
                flipped = True
             
        
class DirectionCalculation(object):
    """Calculate which directions are possible to move in, based on the 6 directions.
    Any combination is fine, as long as it doesn't go back on itself, hence why X, Y 
    and Z have been given two values each, as opposed to just using six values.
    
    Because the code to calculate score will look in one direction then reverse it, 
    the list then needs to be trimmed down to remove any duplicate directions (eg. 
    up/down and upright/downleft are both duplicates)
    """
    
    def __init__(self, C3D):
        direction_group = {}
        direction_group['X'] = 'LR'
        direction_group['Y'] = 'UD'
        direction_group['Z'] = 'FB'
        direction_group[' '] = ' '
        
        #Calculate the edge numbers for each direction
        edges = {'U': list(C3D._range_md),
                 'D': range(C3D._size_squared * (C3D.size - 1), C3D._size_squared * C3D.size),
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
        self.reverse_directions = []
        for direction in opposite_direction:
            
            #Get a list of directions and calculate movement amount
            directions = [list(direction)]
            directions += [[j.replace(i, '') for i in directions[0] for j in direction_group.values() if i in j]]
            direction_movement = sum(move[j] for j in directions[0])
                            
            #Build list of invalid directions
            invalid_directions = [[edges[j] for j in directions[k]] for k in (0, 1)]
            invalid_directions = [join_list(j) for j in invalid_directions]
            
            self.reverse_directions.append((direction_movement, invalid_directions))
        
    
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
    
    def points_nearneighbour(self, player_range, extensive_look=True):
        """Look two moves ahead to detect if someone could get a point.
        Uses the check_for_n_minus_one function from within a loop.
        
        Will return 1 as the second parameter if it has looked up more than a single move.
        """
        
        #Try initial check
        
        match = self.points_immediateneighbour(player_range=player_range)
        if match and not extensive_look:
                return match, True
        near = bool(match)
            
        #For every grid cell, substitute a player into it, then do the check again
        grid = bytearray(self.game.core.grid)
        matches = defaultdict(list)
        for i in self.game.core._range_lg:
            if not self.game.core.grid[i]:
                old_value = grid[i]
                for player in player_range:
                    grid[i] = player
                    match = self.points_immediateneighbour(player_range=player_range, grid=grid)
                    if match:
                        for k, v in match.iteritems():
                            matches[k] += v
                            
                grid[i] = old_value
        
        if matches:
            return matches, near
            
        return defaultdict(list), False
        

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
        
        chance_tactic, chance_ignore, chance_ignore_offset, extensive_look = self.difficulty(difficulty)
        
        total_moves = len([i for i in self.game.core.grid if i])
        self.calculations = 0
        next_moves = []
        
        self.game._ai_text = []
        self.game._ai_state = None
        self.game._ai_move = None
        ai_text = self.game._ai_text.append
        
        #Skip the first few moves since they need the most calculations
        if total_moves >= (self.game.core.size - 2) * len(_range):
            
            #Calculate move
            move_points, is_near = self.points_nearneighbour(player_range=_range, extensive_look=extensive_look)
            ai_text('Urgent: {}'.format(is_near))
            
            #Reduce chance of not noticing n-1 in a row, since n-2 in a row isn't too important
            if is_near:
                chance_ignore /= chance_ignore_offset
                chance_ignore = pow(chance_ignore, pow(total_moves / self.game.core._size_cubed, 0.4))
            
            
            #Chance of things happening
            chance_notice_basic = random.uniform(0, 100) > chance_ignore
            chance_notice_advanced = min(random.uniform(0, 100), random.uniform(0, 100)) > chance_ignore
            chance_new_tactic = random.uniform(0, 100) < chance_tactic
            
            
            #Set which order to do things in
            #OOP of 1 is block first then gain
            order_of_importance = 1 if (is_near or chance_notice_advanced) else -1
                
            #Get max values
            max_points = {}
            for k, v in move_points.iteritems():
                max_length = Counter(v).most_common(1)[0][1]
                max_points[k] = ([i for i in set(v) if v.count(i) == max_length], max_length)
            
            #Get max score
            max_score = {}
            for k, v in self.game.core.score.iteritems():
                if k != player:
                    max_score[k] = v
            max_score = get_max_keys(max_score)
            
            #Set to highest duplicates
            enemy = None
            if chance_notice_advanced:
                
                highest_points = 0
                highest_enemy = defaultdict(list)
                for k, v in max_points.iteritems():
                    if k != player:
                        if v[1] >= highest_points:
                            highest_enemy[k] += v[0]
                            highest_points = v[1]
                            
                highest_score = get_max_keys({k: len(v) for k, v in highest_enemy.iteritems()})
                try:
                    enemy = (highest_score)
                except IndexError:
                    pass
            
            #Set to highest score
            if max_score and enemy is None:
                enemy = max_score
            
            #Set to random player
            else:
                enemy = list(_range)
                del enemy[player - 1]
            
            #Set order
            ai_text('Changed priorities: {}'.format(chance_new_tactic))
            if chance_new_tactic:
                order_of_importance = random.choice((-1, 1))
            order_player = [random.choice(enemy), player][::order_of_importance]
            order_text = ['Blocking opposing player', 'Gaining points'][::order_of_importance]
            
            
            #Predict if other player is trying to trick the AI
            #(or try trick the player if the setup is right)
            #Quite an advanced tactic so chance of not noticing is increased
            if chance_notice_advanced:
                for i in (1, 0):
                    if order_player[i] in max_points and max_points[order_player[i]][1] > 1:
                        next_moves = max_points[order_player[i]][0]
                        self.game._ai_state = 'Forward thinking ({})'.format(order_text[i])
            
            #Make a move based on the current points in the grid
            if move_points and chance_notice_basic and (not next_moves or is_near):
                for i in (1, 0):
                    if move_points[order_player[i]]:
                        next_moves = move_points[order_player[i]]
                        self.game._ai_state = order_text[i]
                    
            #Make a random move determined by number of possible points that can be gained
            elif not self.game._ai_state:
                if not chance_notice_basic:
                    ai_text("AI didn't notice something.")
                self.game._ai_state = False
            
        #Make a semi random placement
        if not self.game._ai_state:
            if not chance_ignore and random.uniform(0, 100) > chance_ignore:
                next_moves = self.points_bestcell(player)
                self.game._ai_state = 'Predictive placement'
            else:
                self.game._ai_state = 'Random placement'
            
        #Make a totally random move
        if not next_moves:
            next_moves = [i for i in self.game.core._range_lg if not self.game.core.grid[i]]
            if self.game._ai_state is None:
                self.game._ai_state = 'Struggling'
                
        ai_text('AI Objective: {}.'.format(self.game._ai_state))
        n = random.choice(next_moves)
        
        ai_text('Potential Moves: {}'.format(next_moves))
        
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
        
        level_data = [(75, 95, 1, False), #Beginner
                      (50, 75, 2, False), #Easy
                      (40, 50, 3, False), #Medium
                      (20, 25, 3, True),  #Hard
                      (0, 0, 1, True)]    #Extreme
                      
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
        
        self.centre = (self.chunk_height / 2) * self.core.size - self.padding / 2
        self.size_x_sm = self.size_x / self.core.size
        self.size_y_sm = self.size_y / self.core.size
        
        self.length_small = self.length / self.core.size
        
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
        chunk_coordinates = [(self.offset[0], self.offset[1] - i * self.chunk_height) for i in self.core._range_sm]
        
        self.line_coordinates = [((self.offset[0] + self.size_x, self.offset[1] + self.centre - self.size_y),
                                  (self.offset[0] + self.size_x, self.offset[1] + self.size_y - self.centre)),
                                 ((self.offset[0] - self.size_x, self.offset[1] + self.centre - self.size_y),
                                  (self.offset[0] - self.size_x, self.offset[1] + self.size_y - self.centre)),
                                 ((self.offset[0], self.offset[1] + self.size_y * 2 - self.centre),
                                  (self.offset[0], self.offset[1] + self.centre))]
        
        for i in self.core._range_sm:

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

    FPS_IDLE = 22
    FPS_MAIN = 30
    FPS_SMOOTH = 120
    TICKS = 120
    WAIT = 60
    WIDTH = 640
    HEIGHT = 960
    
    def __init__(self, C3DGame):
        self.game = C3DGame
        self.move_timer = 200
        
        self.player_colours = [GREEN, LIGHTBLUE, YELLOW, RED, PURPLE]
        random.shuffle(self.player_colours)
    
    def resize_screen(self):
        """Recalculate anything to do with a new width and height."""
        self.mid_point = [self.WIDTH / 2, self.HEIGHT / 2]
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)
        
        #Set length and angle to fit on the screen
        length = 200
        angle = 26
        padding = 4
        angle_limits = (26, 40)
        
        offset = (self.mid_point[0], self.mid_point[1] + self.HEIGHT / 25)
        freeze_edit = False
        while True:
            edited = False
            self.draw = DrawData(self.game.core, length, angle, padding, offset)
            height = self.draw.chunk_height * self.game.core.size
            width = self.draw.size_x * 2
            
            too_small = height < self.HEIGHT * 0.87
            too_tall = height > self.HEIGHT * 0.88
            too_thin = width < self.WIDTH * 0.85
            too_wide = width > self.WIDTH * 0.9
            
            if too_wide:
                if angle < angle_limits[1]:
                    angle += 1
                else:
                    length -= 1
                    freeze_edit = True
                edited = True
                    
            if too_thin:
                if angle > angle_limits[0]:
                    angle -= 1
                    edited = True
                elif not too_tall and not freeze_edit:
                    length += 1
                    edited = True
                
            if too_tall:
                freeze_edit = True
                length -= 1
                edited = True
                
            if not edited:
                break
        
                
        #Set font sizes
        self.text_padding = self.HEIGHT / 96
        self.width_padding = 5
        self.font_lg = pygame.font.Font(self.font_file, self.HEIGHT // 28)
        self.font_lg_size = self.font_lg.render('', 1, BLACK).get_rect()[3]
        self.font_md = pygame.font.Font(self.font_file, self.HEIGHT // 40)
        self.font_md_size = self.font_md.render('', 1, BLACK).get_rect()[3]
        self.font_sm = pygame.font.Font(self.font_file, self.HEIGHT // 45)
        self.font_sm_size = self.font_sm.render('', 1, BLACK).get_rect()[3]
        
        self.score_width = None
        
        #Update surfaces
        self.set_grid_overlay()
        self.set_grid_blocks()
        self.set_game_title()
        

    def end(self):
        """Handle ending the game from anywhere."""
        self.state = None
    
    def set_grid_blocks(self):
    
        self.screen_blocks = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA, 32)
        
        hover = self.temp_data['Hover']
        pending = self.temp_data['PendingMove']
        early = self.temp_data['EarlyClick']
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
        
        core = self.game.core
        for i in core._range_lg:
            if core.grid[i] or i in extra:
                chunk = i // core._size_squared
                base_coordinate = self.draw.relative_coordinates[i % core._size_squared]
                coordinate = (base_coordinate[0] + self.draw.offset[0], 
                    self.draw.offset[1] + chunk * self.draw.chunk_height + self.draw.size_y_sm * 2 - base_coordinate[1])
                
                square = [coordinate,
                          (coordinate[0] + self.draw.size_x_sm,
                           coordinate[1] - self.draw.size_y_sm),
                          (coordinate[0],
                           coordinate[1] - self.draw.size_y_sm * 2),
                          (coordinate[0] - self.draw.size_x_sm,
                           coordinate[1] - self.draw.size_y_sm),
                          coordinate]
                          
                #Player has mouse over square
                block_colour = None
                
                if not core.grid[i]:
                    
                    #Hovering over block
                    if i == extra[0]:
                        block_colour = mix_colour(WHITE, WHITE, self.player_colours[hover[1] - 1])
                        
                    if i == extra[1]:
                        player_colour = self.player_colours[self._player - 1]
                        
                        #Holding down over block
                        if pending[2]:
                            block_colour = mix_colour(BLACK, GREY, WHITE, WHITE, player_colour, player_colour, player_colour, player_colour)
                        
                        #Holding down but moved away
                        else:
                            block_colour = mix_colour(WHITE, WHITE, player_colour)
                    
                    #Hovering over block between turns
                    if i == extra[2]:
                        block_colour = mix_colour(WHITE, WHITE, self.player_colours[early[1] - 1])
                
                #Square is taken by a player
                else:
                    block_colour = self.player_colours[core.grid[i] - 1]
                
                if block_colour is not None:
                    pygame.draw.polygon(self.screen_blocks,
                                        block_colour, square, 0)
                                        
    def set_grid_overlay(self):
        """Draws the grid outline to a surface."""
        
        #Create transparent surface
        self.screen_grid = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA, 32)
              
        #Draw grid
        for line in self.draw.line_coordinates:
            pygame.draw.aaline(self.screen_grid,
                               BLACK, line[0], line[1], 1)
    
    def set_game_time(self):
        """Needs fast updates."""
        self.screen_time = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA, 32)
        if self.temp_data['MoveTimeLeft'] is not None:
            time_left = self.temp_data['MoveTimeLeft'] // self.TICKS + 1
            message = '{} second{}'.format(time_left, 's' if time_left != 1 else '')
            font = self.font_sm.render(message, 1, BLACK)
            size = font.get_rect()[2:]
            self.screen_time.blit(font, ((self.WIDTH - size[0]) / 2, self.text_padding))
    
    def set_game_title(self):
        self.screen_title = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA, 32)
        
        #Display winner
        if self.temp_data['Winner'] is not None:
            if len(self.temp_data['Winner']) == 1:
                message = "Player {} won!".format(self.temp_data['Winner'][0])
            else:
                message = "The game was a draw!"
        
        elif self.game._ai_running:
            message = "Player {} is thinking...".format(self._player)
        
        #Don't instantly switch to player is thinking as it could be a quick click
        elif (self.temp_data['PendingMove'] is None
            or (not self.temp_data['PendingMove'][3] 
                and self.temp_data['PendingMove'][1] > self.frame_data['GameTime'].total_ticks)):
            message = "Player {}'s turn!".format(self._player)
            
        else:
            if self.temp_data['PendingMove'][3]:
                message = "Player {} is moving...".format(self._player)
            else:
                message = "Player {} is thinking...".format(self._player)
        
            
        font = self.font_lg.render(message, 1, BLACK)
        main_size = font.get_rect()[2:]
        self.screen_title.blit(font, ((self.WIDTH - main_size[0]) / 2, self.text_padding * 3))
        
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
        
        point_display = '/'
        
        #Adjust number of points in a row to display
        if self.score_width is None:
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
        self.screen_title.blit(upper_font, (self.width_padding, self.text_padding))
        
        current_height = self.text_padding + upper_size[1]
        for i in range(score1 // self.score_width + 1):
            points = self.score_width if (i + 1) * self.score_width <= score1 else score1 % self.score_width
            lower_font = self.font_lg.render(point_display * points, 1, BLACK)
            lower_size = lower_font.get_rect()[2:]
            self.screen_title.blit(lower_font, (self.width_padding, current_height))
            current_height += lower_size[1] - self.text_padding
        
        #Score 2
        upper_font = self.font_md.render('Player {}'.format(player2), 1, BLACK, self.player_colours[player2 - 1])
        upper_size = upper_font.get_rect()[2:]
        self.screen_title.blit(upper_font, (self.WIDTH - upper_size[0] - self.width_padding, self.text_padding))
        
        current_height = self.text_padding + upper_size[1]
        for i in range(score2 // self.score_width + 1):
            points = self.score_width if (i + 1) * self.score_width <= score2 else score2 % self.score_width
            lower_font = self.font_lg.render(point_display * points, 1, BLACK)
            lower_size = lower_font.get_rect()[2:]
            self.screen_title.blit(lower_font, (self.WIDTH - lower_size[0] - self.text_padding, current_height))
            current_height += lower_size[1] - self.text_padding
        
        #Status message
        if self.temp_data['Winner'] is None and (self.temp_data['Skipped'] or self.temp_data['Flipped']):
            if self.temp_data['Skipped']:
                if self.temp_data['Skipped'] == 2:
                    message = 'Forced move!'
                else:
                    message = 'Switched players!'
                last_player = self.game.previous_player(self._player, self._player_count)
                message += ' (Player {} took too long)'.format(last_player)
            elif self.temp_data['Flipped']:
                message = 'Grid was flipped!'
            font = self.font_md.render(message, 1, (0, 0, 0))
            size = font.get_rect()[2:]
            self.screen_title.blit(font, ((self.WIDTH - size[0]) / 2, self.text_padding * 3 + main_size[1]))
        
    
    def run_ai(self, run=True):
        """Runs the AI in a thread.
        Until _ai_move or _ai_state is not None, it is not completed.
        """
        self.game._ai_move = self.game._ai_state = None
        if run:
            ThreadHelper(self.game.ai.calculate_move, 
                         self._player, 
                         difficulty=self._player_types[self._player - 1], 
                         _range=self._range_players).start()

            
    def play(self, players):
    
        #Initialise screen
        pygame.init()
        
        self.temp_data = {'Hover': None,
                          'PendingMove': None,
                          'EarlyClick': None,
                          'Winner': None,
                          'ShuffleCount': 0,
                          'Flipped': False,
                          'Skipped': False,
                          'MoveTimeLeft': None}
        
        self._player_count = len(players)
        self._range_players = [i + 1 for i in range(self._player_count)]
        self._player = random.choice(self._range_players)
        self._player_types = [i - 1 for i in players]
        #self.game.next_player(self._player, num_players)
        
        #Import the font
        self.font_file = 'Miss Monkey.ttf'
        try:
            pygame.font.Font(self.font_file, 0)
        except IOError:
            print 'Failed to load font'
            return
        
        #Adjust width and height to fit on screen
        screen_height = pygame.display.Info().current_h * 0.9
        height_multiplier = int(screen_height / 16)
        height_ratio = self.WIDTH / self.HEIGHT
        self.HEIGHT = height_multiplier * 16
        self.WIDTH = int(self.HEIGHT * height_ratio)
        
        self.resize_screen()
        self.state = 'Main'
        
        pygame.scrap.init()
        
        GT = GameTime(self.FPS_IDLE, self.TICKS)
        while True:
            with GameTimeLoop(GT) as game_time:
            
                #Store frame specific things so you don't need to call it multiple times
                self.frame_data = {'GameTime': game_time,
                                   'Redraw': False,
                                   'Events': pygame.event.get(),
                                   'Keys': pygame.key.get_pressed(),
                                   'MousePos': pygame.mouse.get_pos(),
                                   'MouseClick': list(pygame.mouse.get_pressed()),
                                   'MouseUse': any(pygame.mouse.get_pressed())}
                if self.frame_data['Keys'][pygame.K_ESCAPE]:
                    return self.end()
                    
                #Handle quitting and resizing window
                if self.state is None:
                    pygame.quit()
                    return
                for event in self.frame_data['Events']:
                    if event.type == pygame.QUIT:
                        self.state = None
                        
                    elif event.type == pygame.VIDEORESIZE:
                        self.WIDTH, self.HEIGHT = event.dict['size']
                        self.resize_screen()
                        self.frame_data['Redraw'] = True
                        
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button <= 3:
                        self.frame_data['MouseUse'] = True
                        self.frame_data['MouseClick'][event.button - 1] = 1
                    
                    if event.type == pygame.MOUSEMOTION:
                        self.frame_data['MouseUse'] = True
                    
                #---MAIN LOOP START---#
                
                
                if self.state == 'Main':
                    self.game_main()
                
                '''
                if self.game._ai_move is not None:
                    print self.game._ai_move, self.game._ai_state
                    self.run_ai(run=False)
                    '''
                
                #---MAIN LOOP END---#
                if game_time.fps:
                    pygame.display.set_caption('{}'.format(game_time.fps))
                    

    def game_main(self):
    
    
        #Count ticks down
        force_end = False
        if self.temp_data['MoveTimeLeft'] is not None:
            self.temp_data['MoveTimeLeft'] -= self.frame_data['GameTime'].ticks
            self.set_game_time()
            if self.temp_data['MoveTimeLeft'] < 0:
                force_end = True + (self.temp_data['PendingMove'] is not None)
                self.set_game_title()
        
        player_type = self._player_types[self._player - 1]
        
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
                next_player = self.game.next_player(self._player, self._player_count)
                
                #Re-activate hover if next player is human
                if self._player_types[next_player - 1] < 0:
                    self.temp_data['Hover'] = [mouse_block_id, next_player]
                    
                else:
                    self.temp_data['Hover'] = None
            
            #Players go
            elif player_type < 0:
                if self.temp_data['PendingMove'] is None:
                    self.temp_data['Hover'] = [mouse_block_id, self._player]
                else:
                    self.temp_data['Hover'] = None
                    
            else:
                self.temp_data['Hover'] = None
                
            self.set_game_title()
            self.set_grid_blocks()
            self.frame_data['Redraw'] = True
            
        
        #Move not yet made
        if (self.temp_data['PendingMove'] is None or not self.temp_data['PendingMove'][3]) and not force_end:
        
            #Human player
            if player_type < 0:
            
                if self.temp_data['MoveTimeLeft'] is None and self.move_timer:
                   self.temp_data['MoveTimeLeft'] = self.TICKS * self.move_timer
                   self.set_game_time()
                    
                #Mouse button clicked
                if self.frame_data['MouseClick'][0]:
                    
                    #Player has just clicked
                    if self.temp_data['PendingMove'] is None:
                        if mouse_block_id is not None and not self.game.core.grid[mouse_block_id]:
                            self.temp_data['PendingMove'] = [mouse_block_id, 
                                                             self.frame_data['GameTime'].total_ticks + self.WAIT, 
                                                             True, 
                                                             False]
                    
                    #Player is holding click over the block
                    elif mouse_block_id == self.temp_data['PendingMove'][0]:
                        self.temp_data['PendingMove'][2] = True
                    
                    #Player is holding click and has moved mouse away
                    elif mouse_block_id != self.temp_data['PendingMove'][0]:
                        self.temp_data['PendingMove'][2] = False
                
                
                #Mouse button released
                elif self.temp_data['PendingMove'] is not None and not self.temp_data['PendingMove'][3]:
                    
                    #If mouse was kept on
                    if self.temp_data['PendingMove'][2]:
                        self.temp_data['PendingMove'][3] = True
                    
                    #If mouse was dragged off
                    else:
                        self.temp_data['PendingMove'] = None
                
                self.set_grid_blocks()
                self.frame_data['Redraw'] = True
                
            
            #Computer player
            elif self.game._ai_running is False:
            
                if self.temp_data['Winner'] is None:
                
                    #Move has not started yet
                    if self.game._ai_move is None:
                        self.run_ai()
                    
                    #Move finished calculating
                    else:
                        self.temp_data['PendingMove'] = [self.game._ai_move, 
                                                         self.frame_data['GameTime'].total_ticks + self.WAIT, 
                                                         True, 
                                                         True]
                                                         
                        self.run_ai(run=False)
                        
                self.set_game_title()
                self.set_grid_blocks()
                self.frame_data['Redraw'] = True
        
        #Commit the move
        else:
            
            #Moved cancelled
            if self.frame_data['MouseClick'][2] and player_type < 0:
                self.temp_data['PendingMove'] = None
                
            elif self.temp_data['PendingMove'] is not None:
                block_id, wait_until, hovering, accept = self.temp_data['PendingMove']
                
                #Cancelled move
                if not accept and not hovering and not self.frame_data['MouseClick'][0]:
                    self.temp_data['PendingMove'] = block_id = wait_until = None
                    
                if (block_id is not None and accept and self.frame_data['GameTime'].total_ticks > wait_until
                    or force_end):
                    self.temp_data['MoveTimeLeft'] = None
                    self.temp_data['PendingMove'] = None
                    self.temp_data['Skipped'] = False
                    self.game.core.grid[block_id] = self._player
                    self._player = self.game.next_player(self._player, self._player_count)
                    
                    self.set_grid_blocks()
                    self.frame_data['Redraw'] = True
                    self.game.core.calculate_score()
                
                    #Shuffle grid
                    self.temp_data['ShuffleCount'] += 1
                    if self.temp_data['ShuffleCount'] >= self.game.shuffle_turns:
                        self.temp_data['ShuffleCount'] = 0
                        self.temp_data['Flipped'] = self.game.core.shuffle()
                    else:
                        self.temp_data['Flipped'] = False
        
            self.set_game_title()
            self.frame_data['Redraw'] = True
            self.temp_data['Winner'] = self.game.check_game_end(self._range_players)
        
        #Copy game state to clipboard
        pygame.scrap.put(pygame.SCRAP_TEXT, base64.b64encode(zlib.compress('{}'.format(self.game.core.grid))))
        
        if force_end:
            if force_end == 1:
                self._player = self.game.next_player(self._player, self._player_count)
            self.temp_data['MoveTimeLeft'] = None
            self.temp_data['Skipped'] = force_end
            self.set_game_title()
                
        #Draw frame
        elif self.frame_data['Redraw']:
            self.screen.fill(BACKGROUND)
            
            grid_dimensions = self.screen_grid.get_size()
            grid_location = [i - j / 2 for i, j in zip(self.mid_point, grid_dimensions)]
            
            self.screen.blit(self.screen_blocks, grid_location)
            self.screen.blit(self.screen_grid, grid_location)
            self.screen.blit(self.screen_title, grid_location)
            if self.temp_data['MoveTimeLeft'] is not None:
                self.screen.blit(self.screen_time, grid_location)
            pygame.display.flip()
            
    
    
c = Connect3DGame(size=4)#.load('eJwtioENADAIwkr/P3pghiFUJaCpdNBx+yGX/Gcyyxq9sYI+CqIAUQ==')
c.play((5, 5))
#GameCore(c).play()
