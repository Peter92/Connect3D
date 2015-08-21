import itertools, operator
from collections import defaultdict
import random


class SwapGridData(object):
    def __init__(self, grid_data):
        self.grid_data = grid_data
        self.grid_size = calculate_grid_size(self.grid_data)
    
    def x(self):
        return join_list(x[::-1] for x in split_string(self.grid_data, self.grid_size))
        
    def y(self):
        return join_list(split_string(self.grid_data, len(self.grid_data)/self.grid_size)[::-1])
    
    def z(self):
        group_split = split_string(self.grid_data, len(self.grid_data)/self.grid_size)
        return join_list(join_list(split_string(x, self.grid_size)[::-1]) for x in group_split)

def calculate_grid_size(grid_data):
    return int(round(pow(len(grid_data), 1.0/3.0), 0))

def split_string(x, n):
    return [x[i:i+n] for i in range(0, len(x), n)]
    
def join_list(x):
    return [j for i in x for j in i]
    

def draw_grid(grid_data):
    grid_size = calculate_grid_size(grid_data)
    k = 0
    for j in range(grid_size):
        print ' '*(grid_size*2) + '_'*(grid_size*4)
        for i in range(grid_size):
            print ' '*(grid_size*2-i*2-1) + '/' + ''.join((' ' + str(grid_data[k+x]).ljust(1) + ' /') for x in range(grid_size))
            k += grid_size
            print ' '*(grid_size*2-i*2-2) + '/' + '___/'*grid_size
            

class CheckGrid(object):
    
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
    some_directions = all_directions
    opposite_direction = all_directions.copy()
    for i in all_directions:
        if i in opposite_direction:
            new_direction = ''
            for j in list(i):
                for k in direction_group.values():
                    if j in k:
                        new_direction += k.replace(j, '')
            opposite_direction.remove(new_direction)
    
    def __init__(self, grid_data):
        
        self.grid_data = grid_data
        self.grid_size = calculate_grid_size(self.grid_data)
        self.grid_size_squared = pow(self.grid_size, 2)
        self.grid_size_cubed = len(self.grid_data)
        
        self.direction_edges = {}
        self.direction_edges['U'] = range(self.grid_size_squared)
        self.direction_edges['D'] = range(self.grid_size_squared*(self.grid_size-1), self.grid_size_squared*self.grid_size)
        self.direction_edges['R'] = [i*self.grid_size+self.grid_size-1 for i in range(self.grid_size_squared)]
        self.direction_edges['L'] = [i*self.grid_size for i in range(self.grid_size_squared)]
        self.direction_edges['F'] = [i*self.grid_size_squared+j+self.grid_size_squared-self.grid_size for i in range(self.grid_size) for j in range(self.grid_size)]
        self.direction_edges['B'] = [i*self.grid_size_squared+j for i in range(self.grid_size) for j in range(self.grid_size)]
        self.direction_edges[' '] = []
        
        self.direction_maths = {}
        self.direction_maths['D'] = pow(self.grid_size, 2)
        self.direction_maths['R'] = 1
        self.direction_maths['F'] = self.grid_size
        self.direction_maths['U'] = -self.direction_maths['D']
        self.direction_maths['L'] = -self.direction_maths['R']
        self.direction_maths['B'] = -self.direction_maths['F']
        self.direction_maths[' '] = 0


    def points(self):

        total_points = defaultdict(int)
        
        all_matches = set()
        
        #Loop through each point
        for starting_point in range(len(self.grid_data)):
            
            current_player = self.grid_data[starting_point]
            
            if current_player:
            
                for i in self.opposite_direction:
                    
                    #Get a list of directions and calculate movement amount
                    possible_directions = [list(i)]
                    possible_directions += [[j.replace(i, '') for i in possible_directions[0] for j in self.direction_group.values() if i in j]]
                    direction_movement = sum(self.direction_maths[j] for j in possible_directions[0])
                    
                    #Build list of invalid directions
                    invalid_directions = [[self.direction_edges[j] for j in possible_directions[k]] for k in (0, 1)]
                    invalid_directions = [join_list(j) for j in invalid_directions]
                    
                    num_matches = 1
                    list_match = [starting_point]
                    
                    #Use two loops for the opposite directions
                    for j in (0, 1):
                        
                        current_point = starting_point
                        
                        while current_point not in invalid_directions[j] and 0 < current_point < self.grid_size_cubed:
                            current_point += direction_movement*int('-'[:j]+'1')
                            if self.grid_data[current_point] == current_player:
                                num_matches += 1
                                list_match.append(current_point)
                            else:
                                break
                    
                    #Add a point if enough matches
                    if num_matches == self.grid_size:
                        
                        list_match = tuple(sorted(list_match))
                        if list_match not in all_matches:
                            all_matches.add(list_match)
                            total_points[current_player] += 1
                            
        
        #Divide points since each combination was counted multiple times
        return {k: v for k, v in total_points.iteritems()}


class PointConversion(object):
    def __init__(self, grid_size, i):
        self.grid_size = grid_size
        self.i = i
        
    def to_3d(self):
        z = self.i / pow(self.grid_size, 2) 
        i = self.i % pow(self.grid_size, 2)
        y = i / grid_size
        x = i % grid_size
        return tuple(i+1 for i in (x, y, z))
    
    def to_int(self):
        x, y, z = self.i
        return (x-1)*pow(self.grid_size, 0) + (y-1)*pow(self.grid_size, 1) + (z-1)*pow(self.grid_size, 2)
        

class Connect3D(object):
    def __init__(self, grid_size=None, raw_data=None):
        if raw_data is not None:
            self.grid_data = list(raw_data)
            self.grid_size = calculate_grid_size(self.grid_data)
            if len(self.grid_data) != pow(self.grid_size, 3):
                self.grid_data = self.grid_data[:pow(self.grid_size, 3)]
        else:
            self.grid_size = grid_size
            self.grid_data = ['' for i in range(pow(grid_size, 3))]
            
    def __repr__(self):
        grid_data_joined = ''.join(str(i).ljust(1) for i in self.grid_data)
        return "Connect3D(raw_data='{}')".format(grid_data_joined)
    
    def update_points(self):
        self.current_points = CheckGrid(self.grid_data).points()
    
    def get_points(self):
        self.update_points()
        for key, value in self.current_points.iteritems():
            print "Player {k}: {v}".format(k=key, v=value)
    
    def _get_max_score(self):
        return max(self.current_points.iteritems(), key=operator.itemgetter(1))[0]
    
    def make_move(self, player, *args):
        if len(args) == 1:
            if not isinstance(args[0], int):
                i = PointConversion(self.grid_size, args[0]).to_int()
            else:
                i = args[0]
        else:
            i = PointConversion(self.grid_size, tuple(args)).to_int()
        if not self.grid_data[i]:
            self.grid_data[i] = player
            return True
        else:
            return False
    
    def draw(self):
        draw_grid(self.grid_data)
    
    def shuffle(self, chance=10):
        
        chance = min(100, chance)
        if chance > 0:
            chance = int(round(300/chance))-1
        else:
            chance = 0
            
        for i in range(3):
            shuffle_list = random.randint(0, chance)
            if shuffle_list == 0:
                self.grid_data = SwapGridData(self.grid_data).x()
            if shuffle_list == 1:
                self.grid_data = SwapGridData(self.grid_data).z()
            if shuffle_list == 2:
                self.grid_data = SwapGridData(self.grid_data).y()

    def play(self, num_players=2, score=3):
        
        player_symbols = ['X', 'O']
        
        i = 0
        self.update_points()
        if self.current_points:
            winning_player = max(self.current_points.iteritems(), key=operator.itemgetter(1))[0]
            print self.current_points
        
        #Remove non player values and find who is the next player
        #self.grid_data = [i if 0<i<num_players+1 else '' for i in self.grid_data]
        number_of_goes = {}
        for i in range(1, num_players+1):
            number_of_goes[i] = self.grid_data.count(i)
        i = min(number_of_goes, key=number_of_goes.get)
        
        while '' in self.grid_data:
            
            self.shuffle()
            self.draw()
            current_player = i % num_players + 1
            self.get_points()
            print 'Player {} move'.format(current_player)
            while True:
                if self.make_move(player_symbols[current_player-1], input()):
                    break
            
            '''  
            try:
                winning_player = max(self.current_points.iteritems(), key=operator.itemgetter(1))[0]
                if self.current_points[winning_player] > score-1:
                    print 'Player {} won!'.format(winning_player)
                    break
            except ValueError:
                pass
                '''
                
            i += 1
        

        
c3d = Connect3D(4)
c3d.play()

