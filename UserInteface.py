import turtle
import math
import operator
turtle.tracer(0, 0)
turtle.clear()
turtle.reset()
turtle.color('black')
turtle.hideturtle()

debug = True
player_colours = {'X': 'red',
                  'O': 'yellow'}

class DrawGrid(object):
    def __init__(self, length, segments, angle, padding=5):
        self.length = length
        self.segments = segments
        self.angle = angle
        self.padding = padding
        self._calculate()

    def _calculate(self):

        self.size_x = self.length * math.cos(math.radians(self.angle))
        self.size_y = self.length * math.sin(math.radians(self.angle))
        self.x_offset = self.size_x / self.segments
        self.y_offset = self.size_y / self.segments
        self.centre = (self.size_y + self.padding / 2) * self.segments
        self.length_small = self.length / self.segments
        
        turtle.up()
        turtle.setpos(0, self.centre)
        turtle.down()

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

    def output(self):
        current_pos = list(range(3))
        index = 0
        grid_segment_range = range(self.segments)
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


        top_corners = [(-self.size_x, self.centre - self.size_y),
                       (self.size_x, self.centre - self.size_y),
                       (0, self.centre - 2 * self.size_y)]
        
        for corner in top_corners:
            turtle.up()
            turtle.setpos(corner)
            turtle.setheading(-90)
            turtle.down()
            turtle.forward((2 * (self.size_y * self.segments - self.size_y)
                            + (self.padding * self.segments - self.padding)))
            
        turtle.update()


class MouseToBlockID(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        if debug:
            turtle.setposition(self.x, self.y)
        self._to_chunk()

    def _to_chunk(self):
        y_offset = grid_main.size_y * 2 + grid_main.padding
        self.y_coordinate = int((grid_main.centre - self.y) / y_offset)
        self.y += y_offset * self.y_coordinate
        self.height = int((grid_main.centre - self.y) / (grid_main.size_y / grid_main.segments))
        self.width = int((self.x + grid_main.size_x) / (grid_main.size_x / grid_main.segments))
        

    def find_x_from_chunk(self):
        """Find spaces that are at the x segment"""
        past_middle = self.width >= grid_main.segments
        
        if not past_middle:
            starting_point = grid_main.segments - self.width
            values = [(starting_point - 1) * grid_main.segments]

            width_addition = 0
            for i in range(starting_point, grid_main.segments):
                n_multiple = grid_main.segments * i
                values.append(n_multiple + width_addition)
                if i < grid_main.segments:
                    values.append(n_multiple + width_addition + 1)
                else:
                    print 'fail'
                    break
                width_addition += 1
                
        else:
            count = 0
            values = []
            while True:
                n_multiple = grid_main.segments * count
                width_addition = self.width - grid_main.segments + count
                if width_addition < grid_main.segments:
                    values.append(n_multiple + width_addition)
                    if width_addition < grid_main.segments - 1:
                        values.append(n_multiple + width_addition + 1)
                else:
                    break
                count += 1
            
        return values

    def find_y_from_chunk(self):
        """Find spaces that are at the y segment"""
        
        height = self.height
        past_middle = height >= grid_main.segments
        if past_middle:
            height = 2 * grid_main.segments - 1 - height
            
        values = []
        count = 0
        while True:
            n_multiple = count * grid_main.segments
            height_addition = height - count
            if height_addition >= 0:
                values.append(n_multiple + height_addition)
                if height_addition >= 1:
                    values.append(n_multiple + height_addition - 1)
            else:
                break
            count += 1
            
        if past_middle:
            values = [pow(grid_main.segments, 2) - i - 1 for i in values]
            
        return values

    def find_possible_blocks(self):
        """Find one or two blocks it could be"""
        
        x_blocks = self.find_x_from_chunk()
        y_blocks = self.find_y_from_chunk()
        return [i for i in x_blocks if i in y_blocks]

    def find_block_coordinates(self):
        matching_blocks = self.find_possible_blocks()
        if not matching_blocks:
            return None
        
        matching_coordinates = {i: grid_main.absolute_coordinates[i]
                                for i in matching_blocks}
        if len(matching_coordinates.keys()) == 1:
            matching_coordinates[-1] = (0, -grid_main.centre)
            
        return sorted(matching_coordinates.items(), key=lambda (k, v): v[1])

    def calculate(self):

        all_blocks = self.find_block_coordinates()
        if all_blocks is None:
            return None
        
        highest_block = all_blocks[1][1]
        line_direction = self.width % 2 == self.height % 2

        
        x1, y1 = (highest_block[0], highest_block[1] - grid_main.y_offset * 2)
        negative = int('-1'[not line_direction:])
        x2, y2 = (x1 + grid_main.x_offset * negative, y1 + grid_main.y_offset)
        sign = (x2 - x1) * (self.y - y1) - (y2 - y1) * (self.x - x1)
        sign *= negative

        if debug:
            turtle.color('red')
            turtle.setposition(x1, y1)
            turtle.setposition(x2, y2)
        
        selected_block = all_blocks[sign > 0][0]
        if selected_block < 0:
            raise IndexError('not on grid')
        return selected_block + self.y_coordinate * pow(grid_main.segments, 2)
        

length = 200
segments = 4
grid_data = ['' for i in range(pow(segments, 3))]

grid_main = DrawGrid(length, segments, 24, padding=5)
grid_main.output()



X=70
Y=40

m = MouseToBlockID(X, Y)
print m.calculate()


turtle.update()

