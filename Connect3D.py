import itertools
import operator
import random
import time
import pygame
import math
from collections import defaultdict
class Connect3DError(Exception):
    pass
'''
Things to do:
marker for disable clicks
'''

BACKGROUND = (250, 250, 255)
LIGHTBLUE = (86, 190, 255)
LIGHTGREY = (200, 200, 200)
GREY = (128, 128, 128)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
SELECTION = {'Default': [WHITE, LIGHTGREY],
             'Hover': [None, BLACK],
             'Waiting': [None, BLACK],
             'Selected': [GREEN, None]}
    
class Connect3D(object):
    """Class to store and use the Connect3D game data.
    The data is stored in a 1D list, but is converted to a 3D representation for the user.
    """
    player_symbols = 'XO'
    default_segments = 4
    default_shuffle_count = 3
    from_string_separator = '/'
    bot_difficulty_default = 'medium'
    
    pygame_overlay_marker = '/'
    pygame_move_marker = '!'
    player_colours = [GREEN, LIGHTBLUE]
    empty_colour = YELLOW
    move_colour = RED
    
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
        """Start or continue a game.
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
                ai_go = SimpleC3DAI(self, self.current_player, players[self.current_player]).calculate_next_move()[0]
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

    def play(self, p1=False, p2=bot_difficulty_default, allow_shuffle=True, end_when_no_points_left=False,
             screen_width=640, screen_height=860,
             default_length=200, default_angle=24):

        self.current_player = int(not self.current_player)

        pygame.init()
        #try:
        #    pygame.init()
        #except NameError:            
        #    return self._old_play(p1, p2, shuffle_x_goes, end_when_no_points_left)
        
        font_file = 'Miss Monkey.ttf'
        try:
            pygame.font.Font(font_file, 0)
        except IOError:
            raise IOError('unable to load font - download from http://www.dafont.com/miss-monkey.font')
            font_file = None
            
            
        convert = CoordinateConvert(screen_width, screen_height)
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Connect 3D')
        background_colour = BACKGROUND
        font_padding = (10, 5)
        padding = 2
        display_offset = (50, 50)
        debug = True

        draw_data = GridDrawData(default_length,
                                 self.segments,
                                 default_angle,
                                 padding = default_angle / self.segments)

        #Flags for held down keys
        flag_dict = {'angle': 0,
                     'size': 0,
                     'overlay': 'options',
                     'player_options': [],
                     'new_game': False,
                     'exit': False,
                     'continue': False,
                     'shuffle': None}

        #How long to wait before accepting a move
        moving_wait = 0.5
        
        #For controlling how the angle and length of grid update
        angle_increment = 0.25
        angle_max = 30
        length_exponential = 1.1
        length_increment = 0.5
        length_multiplier = 0.01
        
        time_current = time.time()
        time_update = 0.01
        
        
        #Initialise various things to update in the code
        reset_colour = None
        block_id = None
        shuffle_count = 0
        moving = False
        moving_start = 0
        moving_override = None
        move_number = 0
        reset_all = False
        was_flipped = False
        ai_turn = None
        ai_message = []
        players = (p1, p2)
        shuffle_data = [allow_shuffle, 3]
        quit_game = False
        
        
        while True:

            #Extra flag for the options menu
            #since it's not in the event loop
            clicked = False
            
            if quit_game:
                return
            
            #Reset loop
            screen.fill(background_colour)
            recalculate = False
            if reset_colour is not None:
                if self.grid_data[reset_colour] == self.pygame_overlay_marker:
                    self.grid_data[reset_colour] = ''
            overlay_used = None

            #Check if no spaces are left
            if '' not in self.grid_data:
                winning_player = self._get_winning_player()
                if len(winning_player) == 1:
                    print 'Player {} won!'.format(winning_player[0])
                else:
                    print 'The game was a draw!'
                pygame.quit()
                return
        
            #Delay each go
            if moving:
                if moving_start < time.time():
                
                    attempted_move = self.make_move(moving[1], moving[0])
                    if attempted_move is not None:

                        move_number += 1
                        self.update_score()
                        shuffle_count += 1
                        if shuffle_count >= shuffle_data[1] and shuffle_data[0]:
                            shuffle_count = 0
                            self.shuffle()
                            was_flipped = True
                        else:
                            was_flipped = False
                    else:
                        self.current_player = int(not self.current_player)
                        print "Invalid move: {}".format(moving[0])
                    moving = False
                    
                else:
                    self.grid_data[moving[0]] = self.pygame_move_marker
                    self.grid_data[moving[0]] = 9 - moving[1]
            
            
            #Run the AI
            if players[self.current_player] is not False:
                if not moving and flag_dict['overlay'] is None:
                    ai_turn, ai_message = SimpleC3DAI(self, self.current_player, difficulty=players[self.current_player]).calculate_next_move()
            else:
                ai_turn = None
            
            #Mouse information
            x_raw, y_raw = pygame.mouse.get_pos()
            x, y = convert.to_pygame(x_raw, y_raw)
            block_id_object = MouseToBlockID(x, y, draw_data)
            block_id = block_id_object.calculate()
            is_taken = True
            if block_id is not None and ai_turn is None:
                is_taken = self.grid_data[block_id] != ''

        
            #Event loop
            event_list = pygame.event.get()
            
            if ai_turn is not None and not moving:
                event_list.append(0)
            
            for event in event_list:

                #Fix for the AI
                try:
                    event_type = event.type
                except AttributeError:
                    event_type = 'mouse'

                if event_type == pygame.QUIT:
                    return

                #Get single key presses
                if event_type == pygame.KEYDOWN:
                    recalculate = True

                    if event.key == pygame.K_ESCAPE:
                        if not flag_dict['overlay']:
                            flag_dict['overlay'] = 'options'
                        else:
                            flag_dict['overlay'] = None
                    
                    if event.key == pygame.K_RIGHTBRACKET:
                        self.segments += 1
                        reset_all = True
                        
                    if event.key == pygame.K_LEFTBRACKET:
                        self.segments -= 1
                        self.segments = max(1, self.segments)
                        reset_all = True

                    if event.key == pygame.K_UP:
                        flag_dict['angle'] = 1

                    if event.key == pygame.K_DOWN:
                        flag_dict['angle'] = -1

                    if event.key == pygame.K_RIGHT:
                        flag_dict['size'] = 1

                    if event.key == pygame.K_LEFT:
                        flag_dict['size'] = -1

                        
                #Get mouse clicks
                if event_type == pygame.MOUSEBUTTONDOWN:
                    clicked = True
                    
                if (event_type == 'mouse' or clicked) and not moving and flag_dict['overlay'] is None:
                    if ai_turn is None and event_type != 'mouse':
                        clicked = True
                        if not is_taken and not players[self.current_player]:
                            moving = (block_id, self.current_player)
                            moving_start = time.time() + moving_wait
                            self.current_player = int(not self.current_player)

                    elif event_type == 'mouse':
                        moving = (ai_turn, self.current_player)
                        moving_start = time.time() + moving_wait
                        self.current_player = int(not self.current_player)

                        
            #Get held down key presses
            key = pygame.key.get_pressed()
            if flag_dict['angle']:
                if not (key[pygame.K_UP] or key[pygame.K_DOWN]):
                    flag_dict['angle'] = 0
                    
                elif time_current < time.time() - time_update:
                    draw_data.angle += angle_increment * flag_dict['angle']
                    recalculate = True
            
            if flag_dict['size']:
                if not (key[pygame.K_LEFT] or key[pygame.K_RIGHT]):
                    flag_dict['size'] = 0
                    
                elif time_current < time.time() - time_update:
                    length_exp = (max(length_increment,
                                     (pow(draw_data.length, length_exponential)
                                      - 1 / length_increment))
                                  * length_multiplier)
                    draw_data.length += length_exp * flag_dict['size']
                    recalculate = True

            if flag_dict['overlay']:
                #if not key[pygame.K_F1]:
                #    flag_dict['overlay'] = None
                overlay_used = flag_dict['overlay']

                
            #Highlight square
            if not is_taken and not moving and not overlay_used:
                self.grid_data[block_id] = self.pygame_overlay_marker
                reset_colour = block_id
            
            #Reinitialise the grid
            if reset_all:
                reset_all = False
                shuffle_data[0] = allow_shuffle
                players = (p1, p2)
                self = Connect3D(self.segments)
                moving = False
                move_number = 0
                reset_colour = None
                recalculate = True
                

            #Recalculate the grid object
            if recalculate:
                draw_data.segments = self.segments
                draw_data.length = max((pow(1 / length_increment, 2)
                                        * draw_data.segments),
                                       draw_data.length,
                                       2)
                draw_data.length = float(draw_data.length)
                draw_data.angle = max(angle_increment, min(89,
                                                           draw_data.angle,
                                                           angle_max))
                draw_data.angle = float(draw_data.angle)
                draw_data._calculate()
                time_current = time.time()


            #Draw coloured squares
            moving_block = None
            for i in self.range_data:
                if self.grid_data[i] != '':
                    chunk = i / self.segments_squared
                    coordinate = list(draw_data.relative_coordinates[i % self.segments_squared])
                    coordinate[1] -= chunk * draw_data.chunk_height
                    
                    square = [coordinate,
                              (coordinate[0] + draw_data.size_x_sm,
                               coordinate[1] - draw_data.size_y_sm),
                              (coordinate[0],
                               coordinate[1] - draw_data.size_y_sm * 2),
                              (coordinate[0] - draw_data.size_x_sm,
                               coordinate[1] - draw_data.size_y_sm),
                              coordinate]

                    if self.grid_data[i] == self.pygame_overlay_marker:
                        block_colour = self.empty_colour
                    elif self.grid_data[i] == self.pygame_move_marker:
                        block_colour = self.move_colour
                    else:
                        j = self.grid_data[i]
                        mix_colour = None
                        if isinstance(j, int) and j > 1:
                            j = 9 - j
                            moving_block = square
                            mix_colour = (255, 128, 128)
                        block_colour = self.player_colours[j]
                        if mix_colour is not None:
                            block_colour = [(block_colour[i] + mix_colour[i]) / 2 for i in range(3)]
                        
                    pygame.draw.polygon(screen,
                                        block_colour,
                                        [convert.to_canvas(*corner)
                                         for corner in square],
                                        0)
                    
                    
            #Draw grid
            for line in draw_data.line_coordinates:
                pygame.draw.aaline(screen,
                                   BLACK,
                                   convert.to_canvas(*line[0]),
                                   convert.to_canvas(*line[1]),
                                   1)
            
            #Draw outline around latest clicked block
            '''
            if moving_block is not None:
                line_coordinates = [[i, (i+1)%4] for i in range(4)]
                for line in line_coordinates:
                    pygame.draw.aaline(screen,
                                       BLUE,
                                       convert.to_canvas(*moving_block[line[0]]),
                                       convert.to_canvas(*moving_block[line[1]]))
                                       '''

            #Draw debug info
            if debug and x is not None and y is not None:
                debug_coordinates = block_id_object.calculate(debug=1)
                if debug_coordinates is not None:
                    if all(i is not None for i in debug_coordinates):
                        pygame.draw.aaline(screen,
                                    RED,
                                    (x_raw, y_raw),
                                    convert.to_canvas(*debug_coordinates[1]),
                                    1)
                        pygame.draw.aaline(screen,
                                    RED,
                                    convert.to_canvas(*debug_coordinates[0]),
                                    convert.to_canvas(*debug_coordinates[1]),
                                    2)

                font = pygame.font.Font(font_file, 16)
                
                #Format the text output
                msg_segments = self.segments
                msg_angle = draw_data.angle
                msg_len = draw_data.length
                x_coordinate = block_id_object.width
                z_coordinate = block_id_object.height
                y_coordinate = block_id_object.y_coordinate
                msg_chunk = (x_coordinate, z_coordinate, y_coordinate)
                msg_x_seg = tuple(block_id_object.find_x_from_chunk())
                msg_y_seg = tuple(block_id_object.find_y_from_chunk())
                msg_possible = tuple(i + y_coordinate * self.segments_squared
                                for i in block_id_object.find_possible_blocks())
                msg_weight = block_id_object.calculate(debug=2)
                
                messages = ['DEBUG INFO',
                            'Segments: {}'.format(msg_segments),
                            'Angle: {}'.format(msg_angle),
                            'Side length: {}'.format(msg_len),
                            'Coordinates: {}'.format((x_raw, y_raw)),
                            'Chunk: {}'.format(msg_chunk),
                            'X Slice: {}'.format(msg_x_seg),
                            'Y Slice: {}'.format(msg_y_seg),
                            'Possible blocks: {}'.format(msg_possible),
                            'Block weight: {}'.format(msg_weight),
                            'Block ID: {}'.format(block_id)]
                font_render = [font.render(i, 1, BLACK)
                               for i in messages]
                font_size = [i.get_rect()[2:] for i in font_render]
                for i in range(len(messages)):
                    message_height = screen_height - sum(j[1] for j in font_size[i:])
                    screen.blit(font_render[i], (0, message_height))

                #Format the AI text output
                ai_message = [i.replace('[', '(').replace(']', ')')[:50] for i in ai_message]
                font_render = [font.render(i, 1, BLACK)
                               for i in ai_message]
                font_size = [i.get_rect()[2:] for i in font_render]

                for i in range(len(ai_message)):
                    message_height = screen_height - sum(j[1] for j in font_size[i:])
                    screen.blit(font_render[i], (screen_width - font_size[i][0], message_height))


            #Format scores
            point_marker = '/'
            font = pygame.font.Font(font_file, 24)
            p1_font_top = font.render('Player 0',
                                      1,
                                      (0, 0, 0),
                                      self.player_colours[0])
            p1_font_size = p1_font_top.get_rect()[2:]
            p2_font_top = font.render('Player 1',
                                      1,
                                      (0, 0, 0),
                                      self.player_colours[1])
            font = pygame.font.Font(font_file, 30)
            p1_font_bottom = font.render(point_marker * self.current_points[0],
                                         1,
                                         (0, 0, 0))
            p2_font_bottom = font.render(point_marker * self.current_points[1],
                                         1,
                                         (0, 0, 0))
            p2_font_size = (p2_font_top.get_rect()[2:],
                            p2_font_bottom.get_rect()[2:])

            screen.blit(p1_font_top,
                        (font_padding[0], font_padding[1]))
            screen.blit(p2_font_top,
                        (screen_width - p2_font_size[0][0] - font_padding[0],
                         font_padding[1]))
            
            screen.blit(p1_font_bottom,
                        (font_padding[0], font_padding[1] + p1_font_size[1]))
            screen.blit(p2_font_bottom,
                        (screen_width - p2_font_size[1][0] - font_padding[0],
                         font_padding[1] + p1_font_size[1]))

            font = pygame.font.Font(font_file, 36)
            player_go_message = "Player {}'s turn!".format(self.current_player)
            player_go_font = font.render(player_go_message,
                                         1,
                                         (0, 0, 0))
            player_go_size = player_go_font.get_rect()[2:]

            screen.blit(player_go_font,
                        ((screen_width - player_go_size[0]) / 2,
                         font_padding[1] * 4))

            if was_flipped:
                font = pygame.font.Font(font_file, 18)
                flipped_message = 'Grid was flipped!'
                flipped_font = font.render(flipped_message,
                                         1,
                                         (0, 0, 0))
                flipped_size = flipped_font.get_rect()[2:]
                screen.blit(flipped_font,
                            ((screen_width - flipped_size[0]) / 2,
                             font_padding[1] * 5 + player_go_size[1]))

            
            #Draw overlay
            if flag_dict['overlay']:
                overlay_width_padding = 50
                overlay_height_padding = 70
                overlay_width = screen_width - overlay_width_padding * 2
                overlay_height = 500
                header_padding = font_padding[1] * 5
                
                #Set font sizes
                font_lg = pygame.font.Font(font_file, 36)
                font_md = pygame.font.Font(font_file, 50)
                font_sm = pygame.font.Font(font_file, 24)
                font_lg_size = font_lg.render('', 1, BLACK).get_rect()[3]
                font_lg_multiplier = 3
                    
                    
                

                #Draw background
                pygame.draw.rect(screen, WHITE, (overlay_width_padding, overlay_height_padding,
                                                 overlay_width, overlay_height))
                pygame.draw.rect(screen, BLACK, (overlay_width_padding, overlay_height_padding,
                                                 overlay_width, overlay_height), 1)

                current_text_height = overlay_height_padding + font_padding[1]

                #Set page titles
                if move_number + bool(moving) and overlay_used == 'options':
                    title_message = 'Options'
                    subtitle_message = ''
                else:
                    title_message = 'Connect 3D'
                    subtitle_message = 'By Peter Hunt'
                 
                #Draw title
                title_text = font_lg.render(title_message, 1, BLACK)
                title_size = title_text.get_rect()[2:]
                screen.blit(title_text, (font_padding[0] + overlay_width_padding,
                                     current_text_height))
                current_text_height += font_padding[1] + title_size[1]
                
                subtitle_text = font_sm.render(subtitle_message, 1, BLACK)
                subtitle_size = subtitle_text.get_rect()[2:]
                screen.blit(subtitle_text, (font_padding[0] + overlay_width_padding,
                                     current_text_height))
                current_text_height += subtitle_size[1]
                
                if subtitle_message:
                    current_text_height += header_padding
                

                if overlay_used == 'options':
                                
                    #Player options
                    p1_text = font_sm.render('Player 0: ', 1, BLACK)
                    p1_size = p1_text.get_rect()[2:]
                    screen.blit(p1_text, (font_padding[0] + overlay_width_padding,
                                          current_text_height))
                    p1_text_height = current_text_height
                    
                    current_text_height += font_padding[1] + p1_size[1]
                    p2_text = font_sm.render('Player 1: ', 1, BLACK)
                    p2_size = p2_text.get_rect()[2:]
                    screen.blit(p2_text, (font_padding[0] + overlay_width_padding,
                                          current_text_height))

                    #Player options - AI selection
                    options = ['Human', 'Beginner', 'Easy', 'Medium', 'Hard', 'Extreme']
                    options_text = [font_sm.render(i, 1, BLACK) for i in options]
                    options_size = [i.get_rect()[2:] for i in options_text]
                    height_list = (p1_text_height, current_text_height)
                    players_unsaved = (p1, p2)
                    option_square_list = defaultdict(list)
                    for j in range(2):

                        #Colour the squares
                        if players_unsaved[j] is False:
                            unsaved_player = -1
                        else:
                            unsaved_player = get_bot_difficulty(players_unsaved[j],
                                                                 _debug=True)
                        if players[j] is False:
                            original_player = -1
                        else:
                            original_player = get_bot_difficulty(players[j], _debug=True)

                        #Draw each square
                        for i in range(len(options)):
                            
                            width_offset = (sum(j[0] for j in options_size[:i])
                                            + font_padding[0] * (i + 2)
                                            + (p2_size if j else p1_size)[0]
                                            + overlay_width_padding)

                            #Set colours
                            option_colours = list(SELECTION['Default'])
                            
                            if i == unsaved_player or unsaved_player < 0 and not i:
                                option_colour = list(SELECTION['Waiting'])
                                rect_colour, text_colour = option_colour
                                if rect_colour is not None:
                                    option_colours[0] = rect_colour
                                if text_colour is not None:
                                    option_colours[1] = text_colour
                                    
                            if i == original_player or original_player < 0 and not i:
                                option_colour = list(SELECTION['Selected'])
                                rect_colour, text_colour = option_colour
                                if rect_colour is not None:
                                    option_colours[0] = rect_colour
                                if text_colour is not None:
                                    option_colours[1] = text_colour
                            
                            if [j, i] == flag_dict['player_options']:
                                flag_dict['player_options'] = []
                                option_colour = list(SELECTION['Hover'])
                                rect_colour, text_colour = option_colour
                                if rect_colour is not None:
                                    option_colours[0] = rect_colour
                                if text_colour is not None:
                                    option_colours[1] = text_colour
                                
                            rect_colour, text_colour = option_colours
                            
                            option_square = (width_offset - padding,
                                             height_list[j] - padding,
                                             options_size[i][0] + padding * 2,
                                             options_size[i][1] + padding)
                            option_square_list[j].append(option_square)

                            pygame.draw.rect(screen, rect_colour, option_square)
                            
                            screen.blit(font_sm.render(options[i], 1, text_colour),
                                        (width_offset, height_list[j]))

                    #Find if player has clicked on a new option
                    for j in option_square_list:
                        for i in range(len(option_square_list[j])):
                            option_square = option_square_list[j][i]
                            if (option_square[0] < x_raw < option_square[0] + option_square[2]
                                and option_square[1] < y_raw < option_square[1] + option_square[3]):

                                player_set = i-1
                                if player_set < 0:
                                    player_set = False
                                flag_dict['player_options'] = [j, i]
                                if clicked:
                                    if not j:
                                        p1 = player_set
                                    else:
                                        p2 = player_set
                                    if not move_number:
                                        players = (p1, p2)


                    current_text_height += header_padding + p2_size[1]

                    #Ask whether to flip the grid
                    flip_grid_text = font_sm.render('Flip grid every 3 goes?',
                                                    1, BLACK)

                    flip_grid_size = flip_grid_text.get_rect()[2:]
                    screen.blit(flip_grid_text,
                                (font_padding[0] + overlay_width_padding,
                                 current_text_height))
                    background_width = screen_width - overlay_width_padding * 2

                    options = ['Yes', 'No']
                    options_text = [font_sm.render(i, 1, BLACK)
                                    for i in options]
                    options_size = [i.get_rect()[2:] for i in options_text]
                    option_square_list = []
                    
                    for i in range(len(options)):
                        width_offset = (sum(j[0] for j in options_size[:i])
                                        + font_padding[0] * (i + 2)
                                        + flip_grid_size[0] + overlay_width_padding)

                        option_square = (width_offset - padding,
                                         current_text_height - padding,
                                         options_size[i][0] + padding * 2,
                                         options_size[i][1] + padding)
                        option_square_list.append(option_square)


                        #Set colours
                        option_colours = list(SELECTION['Default'])

                        if (not i and allow_shuffle
                            or i and not allow_shuffle):
                            option_colour = list(SELECTION['Waiting'])
                            rect_colour, text_colour = option_colour
                            if rect_colour is not None:
                                option_colours[0] = rect_colour
                            if text_colour is not None:
                                option_colours[1] = text_colour

                        if (not i and shuffle_data[0]
                            or i and not shuffle_data[0]):
                            option_colour = list(SELECTION['Selected'])
                            rect_colour, text_colour = option_colour
                            if rect_colour is not None:
                                option_colours[0] = rect_colour
                            if text_colour is not None:
                                option_colours[1] = text_colour

                        if flag_dict['shuffle'] is not None:
                            if (not i and flag_dict['shuffle']
                                or i and not flag_dict['shuffle']):
                                option_colour = list(SELECTION['Hover'])
                                rect_colour, text_colour = option_colour
                                if rect_colour is not None:
                                    option_colours[0] = rect_colour
                                if text_colour is not None:
                                    option_colours[1] = text_colour
                                    
                        rect_colour, text_colour = option_colours

                        pygame.draw.rect(screen, rect_colour, option_square)
                        screen.blit(font_sm.render(options[i],
                                                   1,
                                                   text_colour),
                                    (width_offset, current_text_height))

                    
                    #Find if player has clicked on a new option
                    flag_dict['shuffle'] = None
                    for i in range(len(option_square_list)):
                        option_square = option_square_list[i]
                        if (option_square[0] < x_raw < option_square[0] + option_square[2]
                            and option_square[1] < y_raw < option_square[1] + option_square[3]):
                            flag_dict['shuffle'] = not i
                            if clicked:
                                allow_shuffle = not i
                                if not move_number:
                                    shuffle_data[0] = allow_shuffle
                            
                            
                    current_text_height += header_padding * 2 + flip_grid_size[1]
                    
                    
                    #Tell to restart game
                    if move_number:
                        restart_message = 'Restart game to apply settings.'
                        restart_text = font_sm.render(restart_message,
                                                        1, BLACK)

                        restart_size = restart_text.get_rect()[2:]
                        screen.blit(restart_text,
                                    ((screen_width - restart_size[0]) / 2,
                                     current_text_height))    
                                            

                        current_text_height += header_padding + flip_grid_size[1]

                    
                    #New game button
                    flag_name = 'new_game'
                    button_text = 'New Game' if move_number else 'Start'
                    if self._pygame_button(screen, font_lg, flag_dict, 
                                           flag_name, button_text, x_raw, 
                                           y_raw, clicked, screen_width, 
                                           current_text_height, padding, 
                                           font_lg_multiplier, bool(move_number)):
                        reset_all = True
                        flag_dict['overlay'] = None
                    
                    if move_number:
                        flag_name = 'continue'
                        button_text = 'Continue'
                        if self._pygame_button(screen, font_lg, flag_dict, 
                                               flag_name, button_text, x_raw, 
                                               y_raw, clicked, screen_width, 
                                               current_text_height, padding, 
                                               font_lg_multiplier, -1):
                            flag_dict['overlay'] = None
                        
                    current_text_height += header_padding + font_lg_size
                    
                    #Quit game button
                    flag_name = 'exit'
                    button_text = 'Exit'
                    if self._pygame_button(screen, font_lg, flag_dict, 
                                           flag_name, button_text, x_raw, 
                                           y_raw, clicked, screen_width, 
                                           current_text_height, padding, 
                                           font_lg_multiplier):
                        quit_game = True
                    
                            
                        
                    
            pygame.display.flip()
    
    def _pygame_button(self, screen, font, flag_dict, flag_name, message, x_raw, y_raw,
                       clicked, screen_width, height, padding, multiplier, width_multipler=0):
    
        #Set up text
        text_colour = BLACK if flag_dict[flag_name] else GREY
        text_object = font.render(message, 1, text_colour)
        text_size = text_object.get_rect()[2:]
        
        
        centre_offset = screen_width / 10 * width_multipler
        text_x = (screen_width - text_size[0]) / 2
        if width_multipler > 0:
            text_x += text_size[0] / 2
        if width_multipler < 0:
            text_x -= text_size[0] / 2
        text_x += centre_offset
        
        
        text_square = (text_x - padding * (multiplier + 1),
                       height - padding * multiplier,
                       text_size[0] + padding * (2 * multiplier + 2),
                       text_size[1] + padding * (2 * multiplier - 1))
    
        #pygame.draw.rect(screen, BLACK, text_square)
        screen.blit(text_object, (text_x, height))
        
        flag_dict[flag_name] = False
        
        #Detect if mouse is over it
        if (text_square[0] < x_raw < text_square[0] + text_square[2]
            and text_square[1] < y_raw < text_square[1] + text_square[3]):
            flag_dict[flag_name] = True
            if clicked:
                return True
                
        return False
                
                
                
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
        if 0 <= i < len(self.grid_data) and self.grid_data[i] not in (0, 1) and i is not None:
            self.grid_data[i] = id
            return i
        else:
            return None
            
            
    def shuffle(self, no_shuffle=None):
        """Mirror the grid in the X, Y, or Z axis."""
        
        shuffle_methods = random.sample(range(3), random.randint(0, 2))
        if 0 in shuffle_methods:
            self.grid_data = SwapGridData(self.grid_data).x()
            if self.range_data is not None:
                self.range_data = SwapGridData(self.range_data).x()
        if 1 in shuffle_methods:
            self.grid_data = SwapGridData(self.grid_data).y()
            if self.range_data is not None:
                self.range_data = SwapGridData(self.range_data).y()
        if 2 in shuffle_methods:
            self.grid_data = SwapGridData(self.grid_data).y()
            if self.range_data is not None:
                self.range_data = SwapGridData(self.range_data).y()
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
        self.segments = calculate_segments(self.grid_data)
    
    def x(self):
        """Flip on the X axis.
        
        >>> SwapGridData(range(8)).x()
        [1, 0, 3, 2, 5, 4, 7, 6]
        >>> print Connect3D.from_list(SwapGridData(range(8)).x())
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
        return join_list(x[::-1] for x in split_list(self.grid_data, self.segments))
        
    def y(self):
        """Flip on the Y axis.
        
        >>> SwapGridData(range(8)).y()
        [2, 3, 0, 1, 6, 7, 4, 5]
        >>> print Connect3D.from_list(SwapGridData(range(8)).y())
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
        group_split = split_list(self.grid_data, pow(self.segments, 2))
        return join_list(join_list(split_list(x, self.segments)[::-1]) for x in group_split)
        
    def z(self):
        """Flip on the Z axis.
        
        >>> SwapGridData(range(8)).z()
        [4, 5, 6, 7, 0, 1, 2, 3]
        >>> print Connect3D.from_list(SwapGridData(range(8)).z())
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
        return join_list(split_list(self.grid_data, pow(self.segments, 2))[::-1])
    
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
  
  
class SimpleC3DAI(object):
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
    
    def max_cell_points(self):
        """Get maximum number of points that can be gained from each empty cell,
        that is not blocked by an enemy value.
        """
        max_points = defaultdict(int)
        filled_grid_data = [i if i != '' else self.player for i in self.grid_data]
        for cell_id in range(self.gd_len):
            if cell_id == self.player and self.grid_data[cell_id] == '':
                max_points[cell_id] += self.check_grid(filled_grid_data, cell_id, self.player)

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
                        
                        #print dict(matches)
                        
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
                
                while current_point not in invalid_directions[j] and 0 < current_point < len(grid_data):
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
        output_text = []
        if grid_data_joined_len > (self.C3DObject.segments - 2) * 2:
            
            point_based_move, far_away = SimpleC3DAI(self.C3DObject, self.player_num).look_ahead()
            
            #Reduce chance of not noticing n-1 in a row, since n-2 in a row isn't too important
            if not far_away:
                chance_of_not_noticing /= chance_of_not_noticing_divide
                chance_of_not_noticing = pow(chance_of_not_noticing, pow(grid_data_joined_len / float(len(self.C3DObject.grid_data)), 0.4))
            
            ai_noticed = random.uniform(0, 100) > chance_of_not_noticing
            ai_new_tactic = random.uniform(0, 100) < chance_of_changing_tactic
            
            #Set which order to do things in
            order_of_importance = int('-'[:int(far_away)] + '1')
            if ai_new_tactic:
                output_text.append('AI changed tacic.')
                order_of_importance = random.choice((-1, 1))
            
            move1_player = [self.enemy, self.player][::order_of_importance]
            move1_text = ['Blocking opposing player', 'Gaining points'][::order_of_importance]
            
            state = None
            
            #Make a move based on other points
            if point_based_move and ai_noticed:
                if point_based_move[move1_player[0]]:
                    next_moves = point_based_move[move1_player[0]]
                    state = move1_text[0]
                    
                elif point_based_move[move1_player[1]]:
                    next_moves = point_based_move[move1_player[1]]
                    state = move1_text[1]
            
            #Make a random move determined by number of possible points
            else:
                if not ai_noticed:
                    output_text.append("AI didn't notice something.")
                next_moves = self.max_cell_points()
                state = 'Random placement'

            
            #Make a totally random move
            if not next_moves:
                next_moves = [i for i in range(self.gd_len) if self.grid_data[i] == '']
                if state is None:
                    state = 'Struggling'
        
        else:
            next_moves = [i for i in range(self.gd_len) if self.grid_data[i] == '']
            state = 'Starting'

        output_text.append('AI Objective: {}.'.format(state))
        n = random.choice(next_moves)
        
        output_text.append('Possible Moves: {}'.format(next_moves))
        
        #Find if there are 2 occurances at once, stops people tricking the AI
        occurances = defaultdict(int)
        for i in next_moves:
            occurances[i] += 1
        highest_occurance = max(occurances.iteritems(), key=operator.itemgetter(1))[1]
        next_move = random.choice([k for k, v in occurances.iteritems() if v == highest_occurance])
        
        return next_move, output_text

class MouseToBlockID(object):
    """Converts mouse coordinates into the games block ID.

    The first part is to calculate which level has been clicked, which
    then allows the code to treat the coordinates as level 0. From this
    point, it finds the matching chunks from the new coordinates which
    results in two possible blocks, then it calculates how they are
    conected (highest one is to the left if even+odd, otherwise it's to
    the right), and from that, it's possible to figure out which block
    the cursor is over.
    
    A chunk is a cell of a 2D grid overlaid over the isometric grid.
    Each block is split into 4 chunks, and each chunk overlaps two
    blocks.
    """
    
    def __init__(self, x, y, grid_main):
        self.x = x
        self.y = y
        self.y_original = y
        self.grid_main = grid_main
        self._to_chunk()

    def _to_chunk(self):
        """Calculate which chunk the coordinate is on."""
        y_offset = self.grid_main.size_y * 2 + self.grid_main.padding
        self.y_coordinate = int((self.grid_main.centre - self.y) / y_offset)
        self.y += y_offset * self.y_coordinate
        
        chunk_size_x = self.grid_main.size_x / self.grid_main.segments
        chunk_size_y = self.grid_main.size_y / self.grid_main.segments
        self.height = int((self.grid_main.centre - self.y) / chunk_size_y)
        self.width = int((self.x + self.grid_main.size_x + chunk_size_x) / chunk_size_x) -1
        

    def find_x_from_chunk(self):
        """Find block IDs that are on the x segment"""
        past_middle = self.width >= self.grid_main.segments
        if not past_middle:
            starting_point = self.grid_main.segments - self.width
            values = [(starting_point - 1) * self.grid_main.segments]

            width_addition = 0
            for i in range(starting_point, self.grid_main.segments):
                n_multiple = self.grid_main.segments * i
                values.append(n_multiple + width_addition)
                if 0 < i < self.grid_main.segments:
                    values.append(n_multiple + width_addition + 1)
                else:
                    break
                width_addition += 1
                
        else:
            count = 0
            values = []
            while True:
                n_multiple = self.grid_main.segments * count
                width_addition = self.width - self.grid_main.segments + count
                if width_addition < self.grid_main.segments:
                    values.append(n_multiple + width_addition)
                    if width_addition < self.grid_main.segments - 1:
                        values.append(n_multiple + width_addition + 1)
                else:
                    break
                count += 1
            
        return values

    def find_y_from_chunk(self):
        """Find block IDs that are on the y segment"""
        
        height = self.height
        past_middle = height >= self.grid_main.segments
        if past_middle:
            height = 2 * self.grid_main.segments - 1 - height
            
        values = []
        count = 0
        while True:
            n_multiple = count * self.grid_main.segments
            height_addition = height - count
            if height_addition >= 0:
                values.append(n_multiple + height_addition)
                if height_addition >= 1:
                    values.append(n_multiple + height_addition - 1)
            else:
                break
            count += 1
            
        if past_middle:
            values = [pow(self.grid_main.segments, 2) - i - 1 for i in values]
            
        return values

    def find_possible_blocks(self):
        """Combine the block IDs to find the 1 or 2 matching ones."""
        
        x_blocks = self.find_x_from_chunk()
        y_blocks = self.find_y_from_chunk()
        if self.y_coordinate >= self.grid_main.segments:
            return []
        return [i for i in x_blocks if i in y_blocks]

    def find_block_coordinates(self):
        """Calculate the coordinates of the block IDs, or create a fake
        block if one is off the edge.
        Returns a list sorted by height.

        If only one value is given for which blocks are in the chunk, that
        means the player is on the edge of the board. By creating a fake
        block off the side of the board, it allows the coorect maths to be
        done without any modification.
        """
        matching_blocks = self.find_possible_blocks()
        if not matching_blocks:
            return None
        
        matching_coordinates = {i: self.grid_main.relative_coordinates[i]
                                for i in matching_blocks}

        #Create new value to handle 'off edge' cases
        if len(matching_coordinates.keys()) == 1:
            
            single_coordinate = matching_coordinates[matching_blocks[0]]
            
            new_location = (0, -self.grid_main.centre)

            #Workaround to handle the cases in the upper half
            if self.height < self.grid_main.segments:
                
                top_row_right = range(1, self.grid_main.segments)
                top_row_left = [i * self.grid_main.segments
                                for i in range(1, self.grid_main.segments)]
                if self.width >= self.grid_main.segments:
                    top_row_right.append(0)
                else:
                    top_row_left.append(0)

                
                if matching_blocks[0] in top_row_left:
                    new_location = (single_coordinate[0] - self.grid_main.x_offset,
                                    single_coordinate[1] + self.grid_main.y_offset)

                elif matching_blocks[0] in top_row_right:
                    new_location = (single_coordinate[0] + self.grid_main.x_offset,
                                    single_coordinate[1] + self.grid_main.y_offset)
            
            matching_coordinates[-1] = new_location
            
        return sorted(matching_coordinates.items(), key=lambda (k, v): v[1])

    
    def calculate(self, debug=0):
        """Calculate which block ID the coordinates are on.
        This calculates the coordinates of the line between the two
        blocks, then depending on if a calculation results in a positive
        or negative number, it's possible to detect which block it falls
        on.

        By returning the (x1, y1) and (x2, y2) values, they can be linked
        with turtle to see it how it works under the hood.
        """
        all_blocks = self.find_block_coordinates()
        if all_blocks is None:
            return None
        
        highest_block = all_blocks[1][1]
        line_direction = self.width % 2 == self.height % 2
        if self.grid_main.segments % 2:
            line_direction = not line_direction
        #print self.width, self.height
        
        x1, y1 = (highest_block[0],
                  highest_block[1] - self.grid_main.y_offset * 2)
        negative = int('-1'[not line_direction:])
        x2, y2 = (x1 + self.grid_main.x_offset * negative,
                  y1 + self.grid_main.y_offset)

        sign = (x2 - x1) * (self.y - y1) - (y2 - y1) * (self.x - x1)
        sign *= negative

        #Return particular things when debugging
        if debug == 1:
            return (x1, y1), (x2, y2)
        if debug == 2:
            return sign

        selected_block = all_blocks[sign > 0][0]

        #If extra block was added, it was -1, so it is invalid
        if selected_block < 0:
            return None

        
        return selected_block + self.y_coordinate * pow(self.grid_main.segments, 2)


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
    
    def __init__(self, length, segments, angle, padding=5):
        self.length = length
        self.segments = segments
        self.angle = angle
        self.padding = padding
        self._calculate()

    def _calculate(self):
        """Perform the main calculations on the values in __init__.
        This allows updating any of the values, such as the isometric
        angle, without creating a new class."""
        
        self.size_x = self.length * math.cos(math.radians(self.angle))
        self.size_y = self.length * math.sin(math.radians(self.angle))
        self.x_offset = self.size_x / self.segments
        self.y_offset = self.size_y / self.segments
        self.chunk_height = self.size_y * 2 + self.padding
        
        self.centre = (self.chunk_height / 2) * self.segments - self.padding / 2
        self.size_x_sm = self.size_x / self.segments
        self.size_y_sm = self.size_y / self.segments

        #self.segments_sq = pow(self.segments, 2)
        #self.grid_data_len = pow(self.segments, 3)
        #self.grid_data_range = range(self.grid_data_len)

        
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
        chunk_coordinates = [(0, - i * self.chunk_height) for i in range(self.segments)]

        self.line_coordinates = [((self.size_x, self.centre - self.size_y),
                                  (self.size_x, self.size_y - self.centre)),
                                 ((-self.size_x, self.centre - self.size_y),
                                  (-self.size_x, self.size_y - self.centre)),
                                 ((0, self.centre - self.size_y * 2),
                                  (0, -self.centre))]

        for i in range(self.segments):

            chunk_height = -i * self.chunk_height

            self.line_coordinates += [((self.size_x, self.centre + chunk_height - self.size_y),
                                       (0, self.centre + chunk_height - self.size_y * 2)),
                                      ((-self.size_x, self.centre + chunk_height - self.size_y),
                                       (0, self.centre + chunk_height - self.size_y * 2))]

            for coordinate in self.relative_coordinates:
                
                start = (coordinate[0], chunk_height + coordinate[1])
                self.line_coordinates += [(start,
                                           (start[0] + self.size_x_sm, start[1] - self.size_y_sm)),
                                          (start,
                                           (start[0] - self.size_x_sm, start[1] - self.size_y_sm))]

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


if __name__ == '__main__':
    C3D = Connect3D()
    C3D.play()
