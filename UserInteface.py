mport turtle
import math
import operator
import pygame
import time

debug = True

class DrawGrid(object):
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
        self.centre = (self.size_y + self.padding / 2) * self.segments
        self.length_small = self.length / self.segments
        
        self.absolute_coordinates = []
        position = (0, self.centre)
        for j in range(self.segments):
            checkpoint = position
            for i in range(self.segments):
                self.absolute_coordinates.append(position)
                position = (position[0] + self.x_offset,
                            position[1] - self.y_offset)
            position = (checkpoint[0] - self.x_offset,
                        checkpoint[1] - self.y_offset)

    def square(self, colour=None, player=None):
        """Draw an isometric square, and fill if required."""

        if player:
            colour = player_colours[player]
        if colour:
            turtle.begin_fill()
            turtle.color(colour)
            
        turtle.right(self.angle)
        turtle.forward(self.length_small)
        turtle.right(180 - self.angle * 2)
        turtle.forward(self.length_small)
        turtle.right(self.angle * 2)
        turtle.forward(self.length_small)
        turtle.right(180 - self.angle * 2)
        turtle.forward(self.length_small)
        turtle.setheading(0)
        
        if colour is not None:
            turtle.end_fill()
            turtle.color('black')
            self.square()

    def output(self, grid_data):
        """Draw the main grid with turtle."""
        
        current_pos = list(range(3))
        index = 0
        grid_segment_range = range(self.segments)

        turtle.tracer(0, 0)
        turtle.clear()
        turtle.reset()
        turtle.color('black')
        turtle.hideturtle()

        #Move starting point upwards to centre the grid
        turtle.up()
        turtle.setpos(0, self.centre)
        turtle.down()

        #Draw the main grid
        for k in grid_segment_range:
            current_pos[0] = turtle.pos()
            for j in grid_segment_range:
                current_pos[1] = turtle.pos()
                for i in grid_segment_range:
                    current_pos[2] = turtle.pos()
                    self.square(player = grid_data[index])
                    index += 1
                    turtle.up()
                    turtle.setpos(current_pos[2][0] + self.x_offset,
                                  current_pos[2][1] - self.y_offset)
                    turtle.down()
                turtle.up()
                turtle.setpos(current_pos[1][0] - self.x_offset,
                              current_pos[1][1] - self.y_offset)
                turtle.down()
            turtle.up()
            turtle.setpos(current_pos[0][0],
                          current_pos[0][1] - self.size_y * 2 - self.padding)
            turtle.down()

        #Draw the sides
        top_corners = [(-self.size_x, self.centre - self.size_y),
                       (0, self.centre - 2 * self.size_y),
                       (self.size_x, self.centre - self.size_y)]
        
        for corner in top_corners:
            turtle.up()
            turtle.setpos(corner)
            turtle.setheading(-90)
            turtle.down()
            turtle.forward((2 * (self.size_y * self.segments - self.size_y)
                            + (self.padding * self.segments - self.padding)))
            


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
        self.height = int((self.grid_main.centre - self.y) / (self.grid_main.size_y / self.grid_main.segments))
        self.width = int((self.x + self.grid_main.size_x) / (self.grid_main.size_x / self.grid_main.segments))

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
                if i < self.grid_main.segments:
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
        
        matching_coordinates = {i: self.grid_main.absolute_coordinates[i]
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
                if self.x > 0:
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
    
    def calculate(self, debug=False):
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

        if debug:
            return (x1, y1), (x2, y2)

        selected_block = all_blocks[sign > 0][0]

        #If extra block was added, it was -1, so it is invalid
        if selected_block < 0:
            return None

        
        return selected_block + self.y_coordinate * pow(self.grid_main.segments, 2)

def turtle_coordinates_to_pygame(x, y, width, height):
        centre = (width / 2, height / 2)
        x = - centre[0] + x
        y = centre[1] - y
        return (x, y)


class MainGame(object):

    def main(self, length, segments):
        # Initialise screen
        
        pygame.init()
        width = 480
        height = 840
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Connect 3D')
        debug = True
        
        angle = 22
        grid_data = ['' for i in range(pow(segments, 3))]
        grid_main = DrawGrid(length, segments, angle, padding=5)
        
        angle_increment = 0.5
        length_increment = 1

        angle_flag_up = False
        angle_flag_down = False
        size_flag_up = False
        size_flag_down = False
        update_grid_time = time.time()
        update_grid_max_time = 0.25
        
        reset_colour = None
        block_id = None
        while True:
            
            flags = any((angle_flag_up,
                         angle_flag_down,
                         size_flag_up,
                         size_flag_down))

            key = pygame.key.get_pressed()
            key_pressed = any(key)

            if reset_colour is not None:
                if grid_data[reset_colour] == '/':
                    grid_data[reset_colour] = ''
                    
            #Mouse information
            x, y = pygame.mouse.get_pos()
            relative = turtle_coordinates_to_pygame(x, y, width, height)
            block_id = MouseToBlockID(relative[0], relative[1], grid_main).calculate()
            is_taken = True
            if block_id is not None:
                is_taken = grid_data[block_id] != ''
                if not is_taken:
                    grid_data[block_id] = '/'
                    reset_colour = block_id

            #Event loop
            for event in pygame.event.get():

                #Get mouse clicks
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if not is_taken:
                        grid_data[block_id] = 'X'
                        reset_colour = None

                #Get single key presses
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    
                    if event.key == pygame.K_RIGHTBRACKET:
                        grid_main.segments += 1
                        grid_main._calculate()
                        grid_data = ['' for i in range(pow(grid_main.segments, 3))]
                        reset_colour = None
                        
                    if event.key == pygame.K_LEFTBRACKET:
                        grid_main.segments -= 1
                        grid_main._calculate()
                        grid_data = ['' for i in range(pow(grid_main.segments, 3))]
                        reset_colour = None
                        
                
            #Get continuous key presses
            if key_pressed or flags:
                
                #Unset the flags
                if angle_flag_up and not key[pygame.K_UP]:
                    angle_flag_up = False
                if angle_flag_down and not key[pygame.K_DOWN]:
                    angle_flag_down = False
                if size_flag_down and not key[pygame.K_LEFT]:
                    size_flag_down = False
                if size_flag_up and not key[pygame.K_RIGHT]:
                    size_flag_up = False

                time_to_update = update_grid_time > update_grid_max_time


                if (time_to_update and angle_flag_up
                    or key_pressed and key[pygame.K_UP]):
                    angle_flag_up = True
                    grid_main.angle += angle_increment
                    grid_main.angle = min(89, grid_main.angle)

                if (time_to_update and angle_flag_down
                    or key_pressed and key[pygame.K_DOWN]):
                    angle_flag_down = True
                    grid_main.angle -= angle_increment
                    grid_main.angle = max(angle_increment, grid_main.angle)

                if (time_to_update and size_flag_up
                    or key_pressed and key[pygame.K_RIGHT]):
                    size_flag_up = True
                    grid_main.length += length_increment * segments

                if (time_to_update and size_flag_down
                    or key_pressed and key[pygame.K_LEFT]):
                    size_flag_down = True
                    grid_main.length -= length_increment * segments
                    grid_main.length = max(length_increment * segments, grid_main.length)

                if time_to_update:
                    update_grid_time = time.time()

            #Smooth grid edges
            if not flags:
                if grid_main.length > grid_main.segments and grid_main.length % grid_main.segments:
                    grid_main.length = int(grid_main.length / grid_main.segments) * grid_main.segments
            
                if grid_main.angle > angle_increment and grid_main.angle % angle_increment:
                    grid_main.angle = int(grid_main.angle / angle_increment) * angle_increment


            grid_main._calculate()
            grid_main.output(grid_data)

                
            if debug and relative:
                debug_point_1, debug_point_2 = None, None
                line_coordinates = MouseToBlockID(relative[0], relative[1], grid_main).calculate(debug)
                if line_coordinates:
                    debug_point_1, debug_point_2 = line_coordinates

                turtle.color('grey')
                turtle.setposition(*relative)
                if debug_point_1 is not None and debug_point_2 is not None:
                    turtle.color('red')
                    turtle.setposition(debug_point_1)
                    turtle.setposition(debug_point_2)
                    
            turtle.update()
