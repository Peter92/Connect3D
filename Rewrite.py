from __future__ import division
from collections import defaultdict, Counter
from operator import itemgetter
import random
import base64
try:
    import pygame
except ImportError:
    pygame = None
    
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
        self._shuffle_order = []
        
        #Main parts
        self.grid = bytearray(0 for _ in self._range_lg)
        self.flip = FlipGrid(self)
        self.directions = DirectionCalculation(self)
        self.calculate_score()

    def __repr__(self):
        output = (bytearray([self._player]) + bytearray([self.num_players]) + self.grid)
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
        if not level:
            return
            
        all_flips = (self.flip.fx, self.flip.fy, self.flip.fz, 
                     self.flip.rx, self.flip.ry, self.flip.rz, self.flip.reverse)
        max_shuffles = level * 3
        shuffles = random.sample(range(max_shuffles), random.randint(0, max_shuffles - 1))
        shuffles.append(len(all_flips) - 1)
        
        
        #Perform shuffle
        for i in shuffles:
            self.grid, operation = all_flips[i](self.grid)
            self._shuffle_order.append(operation)
        
        #Clean shuffle order
        new_reverse = self._shuffle_order.count('r') % 2
        new_fx = self._shuffle_order.count('fx') % 2
        new_fy = self._shuffle_order.count('fy') % 2
        new_fz = self._shuffle_order.count('fz') % 2
        new_rx = (self._shuffle_order.count('rx1') - self._shuffle_order.count('rx2')) % 4
        new_ry = (self._shuffle_order.count('ry1') - self._shuffle_order.count('ry2')) % 4
        new_rz = (self._shuffle_order.count('rz1') - self._shuffle_order.count('rz2')) % 4
        self._shuffle_order = (['r'] * new_reverse +
                               ['fx'] * new_fx +
                               ['fy'] * new_fy +
                               ['fz'] * new_fz +
                               ['rx1'] * new_rx +
                               ['ry1'] * new_ry +
                               ['rz1'] * new_rz)
        

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


    def _point_score(self, id, player=None):
        """Find how many points are gained from a cell, and return the row hashes.
        
        Set an optional player value to force the first value to be that player.
        """
        
        row_hashes = set()
        if player is None:
            player = self.grid[id]
        if not player:
            return row_hashes
        
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
                row_hashes.add(hash(tuple(sorted(list_match))))
                
        return row_hashes
        

class Connect3DGame(object):
    DEFAULT_PLAYERS = 2
    DEFAULT_SHUFFLE_TURNS = 3
    
    def __init__(self, num_players=None, shuffle_level=None, shuffle_turns=None, size=None):
        self.num_players = self.DEFAULT_PLAYERS if num_players is None else max(1, num_players)
        self.shuffle_turns = self.DEFAULT_SHUFFLE_TURNS if shuffle_turns is None else max(0, shuffle_turns)
        self.core = Connect3D(size=size, shuffle_level=shuffle_level)
        self.ai = ArtificialIntelligence(self)
        self._range_players = [i + 1 for i in range(self.num_players)]
        self._ai_text = []
        
        try:
            self._player
        except AttributeError:
            self._player = random.randint(1, self.num_players)
        
    def next_player(self):
        if self.num_players < 2:
            return
        self._player += 1
        if self._player != self.num_players:
            self._player %= self.num_players
    
    def previous_player(self):
        if self.num_players < 2:
            return
        self._player -= 1
        if self._player == 0:
            self._player = self.num_players
        
    @classmethod
    def load(cls, data):
        """Calculate the number of segments and split the data into the correct sections.
        
        Parameters:
            x (string): Data to split. 
                Must be a base 64 encoded bytearray.
                Acceptable inputs:
                    bytearray(grid_data)
                    bytearray(current_player + num_players + grid_data)
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
    
    def play(self, ai):
        
        #Validate AI input
        if ai is not None:
            try:
                for k, v in ai.iteritems():
                    try:
                        self.ai.difficulty(v)
                    except IndexError:
                        raise IndexError('ai difficulty must be between 0-4')
                    if k not in self._range_players:
                        raise IndexError('ai must be set to a valid player')
            except AttributeError:
                raise ValueError('ai must be input as {player: level} format')
                
        if pygame:
            return
        
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
            if self._ai_text:
                print self._ai_text
            self._ai_text = []
            
            #Check if any points are left to gain
            points_left = True
            end_early = True
            if end_early:
                potential_points = {j + 1: Connect3D(self.core.size).set_grid([j + 1 if not i else i for i in self.core.grid]).score for j in range(self.num_players)}
                if any(self.core.score == potential_points[player + 1] for player in range(self.num_players)):
                    points_left = False
                    
            #Check if no spaces are left
            if 0 not in self.core.grid or not points_left:
                winning_player = get_max_keys(self.core.score)
                if len(winning_player) == 1:
                    print 'Player {} won!'.format(winning_player[0])
                else:
                    print 'The game was a draw!'
                    
                #Ask to play again and check if answer is a variant of 'yes' or 'ok'
                print 'Play again?'
                play_again = raw_input().lower()
                if any(i in play_again for i in ('y', 'k')):
                    self.core = Connect3D(size=self.core.size, shuffle_level=self.core.shuffle_level)
                    return self.play()
                else:
                    return
            
            
            print "Player {}'s turn".format(self._player)
            if self._player in ai:
                new_go = self.ai.calculate_move(self._player, ai[self._player])
                self.core.grid[new_go] = self._player
                self.next_player()
                
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
                        self.core.grid[new_go] = self._player
                        self.next_player()
                        break
                    if new_go > max_go:
                        print 'input must be between 0 and {}'.format(max_go)
                    elif self.core.grid[new_go]:
                        print 'input is taken'
                    else:
                        print 'unknown error with input'
            
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
        
        try:
            self.calculations += self.game.core._size_cubed
        except AttributeError:
            pass
            
        self._temp_core.grid = grid
        return len(self._temp_core._point_score(cell_id, player))
        
        
    def points_bestcell(self, player):
        """Get maximum number of points that can be gained from each empty cell,
        that is not blocked by an enemy value.
        """
        max_points = defaultdict(int)
        filled_grid = [i if i else player for i in self.game.core.grid]
        for cell_id in self.game.core._range_lg:
            if filled_grid[cell_id] == player and not self.game.core.grid[cell_id]:
                max_points[cell_id] = self.check_cell(cell_id, filled_grid)
        
        return get_max_keys(max_points)

    def points_immediateneighbour(self, grid=None, player=None):
        """Find all places where anyone has n-1 points in a row, by substituting
        in a point for each player in every cell.
        
        Parameters:
            grid_data (list or None, optional): Pass in a custom grid_data, 
                leave as None to use the Connect3D one.
        """
        if grid is None:
            grid = self.game.core.grid
        
        matches = defaultdict(list)
        for cell_id in self.game.core._range_lg:
            if not grid[cell_id]:
                for player in self.game._range_players:
                    if self.check_cell(cell_id, grid, player):
                        matches[player].append(cell_id)
        
        return matches
    
    def points_nearneighbour(self, extensive_look=True):
        """Look two moves ahead to detect if someone could get a point.
        Uses the check_for_n_minus_one function from within a loop.
        
        Will return 1 as the second parameter if it has looked up more than a single move.
        """
        
        #Try initial check
        if not extensive_look:
            match = self.points_immediateneighbour()
            if match:
                return match, True
            
        #For every grid cell, substitute a player into it, then do the check again
        grid = bytearray(self.game.core.grid)
        matches = defaultdict(list)
        for i in self.game.core._range_lg:
            if not self.game.core.grid[i]:
                old_value = grid[i]
                for player in self.game._range_players:
                    grid[i] = player
                    match = self.points_immediateneighbour(grid)
                    if match:
                        for k, v in match.iteritems():
                            matches[k] += v
                            
                grid[i] = old_value
        
        if matches:
            return matches, False
            
        return defaultdict(list), False
        

    def calculate_move(self, player, difficulty=None):
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
        
        
        chance_tactic, chance_ignore, chance_ignore_offset = self.difficulty(difficulty)
        
        total_moves = len([i for i in self.game.core.grid if i])
        self.calculations = 0
        self.game._ai_text = []
        ai_text = self.game._ai_text.append
        state = None
        next_moves = []
        
        if total_moves >= (self.game.core.size - 2) * 2 or True:
            
            #Calculate move
            move_points, is_near = self.points_nearneighbour()
            print move_points
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
                enemy = list(self.game._range_players)
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
                    if order_player[i] in max_points:
                        next_moves = max_points[order_player[i]][0]
                        state = 'Forward thinking ({})'.format(order_text[i])
            
            #Make a move based on the current points in the grid
            if move_points and chance_notice_basic and (not next_moves or is_near):
                for i in (1, 0):
                    if move_points[order_player[i]]:
                        next_moves = move_points[order_player[i]]
                        state = order_text[i]
                    
            #Make a random move determined by number of possible points that can be gained
            elif not state:
                if not chance_notice_basic:
                    ai_text("AI didn't notice something.")
                state = False
            
            #Make a semi random placement
            if not state:
                if not chance_ignore and random.uniform(0, 100) > chance_ignore:
                    next_moves = self.points_bestcell(player)
                    state = 'Predictive placement'
                else:
                    state = 'Random placement'
            
        if state is None:
            state = 'Starting'
            
        #Make a totally random move
        if not next_moves:
            next_moves = [i for i in self.game.core._range_lg if not self.game.core.grid[i]]
            if state is None:
                state = 'Struggling'
                
        ai_text('AI Objective: {}.'.format(state))
        n = random.choice(next_moves)
        
        ai_text('Potential Moves: {}'.format(next_moves))
        
        next_move = random.choice(next_moves)
        
        ai_text('Chosen Move: {}'.format(next_move))
        ai_text('Calculations: {}'.format(self.calculations + 1))
        
        print state, self.game._ai_text
        return next_move
        

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
            
        return [(75, 95, 1),
                (50, 75, 2),
                (40, 50, 3),
                (20, 25, 3),
                (0, 0, 1)][level]
        
        
c = Connect3DGame(shuffle_level=False)
c.play(ai={2:4})
