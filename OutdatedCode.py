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
        

    def find_x_slice(self):
        """Find block IDs that are on the x segment"""
        past_middle = self.width >= self.grid_main.segments
        
        values = []
        if self.width >= self.grid_main.segments:
        
            count = 0
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
        
        elif self.width >= 0:
        
            starting_point = self.grid_main.segments - self.width
            values.append((starting_point - 1) * self.grid_main.segments)

            width_addition = 0
            for i in range(starting_point, self.grid_main.segments):
                n_multiple = self.grid_main.segments * i
                values.append(n_multiple + width_addition)
                if 0 < i < self.grid_main.segments:
                    values.append(n_multiple + width_addition + 1)
                else:
                    break
                width_addition += 1
        
            
        return values

    def find_y_slice(self):
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

    def find_overlap(self):
        """Combine the block IDs to find the 1 or 2 matching ones."""
        
        x_blocks = self.find_x_slice()
        y_blocks = self.find_y_slice()
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
        matching_blocks = self.find_overlap()
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
        
#Old debug for pygame
def _draw_debug(self, block_data):
    """Show the debug information."""

    mouse_data = pygame.mouse.get_pos()
    x, y = self.to_pygame(*mouse_data)
    
    debug_coordinates = block_data['object'].calculate(debug=1)
    if debug_coordinates is not None:
        if all(i is not None for i in debug_coordinates):
            pygame.draw.aaline(self.screen,
                        RED,
                        pygame.mouse.get_pos(),
                        self.to_canvas(*debug_coordinates[1]),
                        1)
            pygame.draw.line(self.screen,
                        RED,
                        self.to_canvas(*debug_coordinates[0]),
                        self.to_canvas(*debug_coordinates[1]),
                        2)

    possible_blocks = block_data['object'].find_overlap()
    
    y_mult = str(block_data['object'].y_coordinate * self.C3DObject.segments_squared)
    if y_mult[0] != '-':
        y_mult = '+{}'.format(y_mult)
    info = ['DEBUG INFO',
            'FPS: {}'.format(int(round(self.clock.get_fps(), 0))),
            'Segments: {}'.format(self.C3DObject.segments),
            'Angle: {}'.format(self.draw_data.angle),
            'Side length: {}'.format(self.draw_data.length),
            'Coordinates: {}'.format(mouse_data),
            'Chunk: {}'.format((block_data['object'].width,
                                block_data['object'].height,
                                block_data['object'].y_coordinate)),
            'X Slice: {}'.format(block_data['object'].find_x_slice()),
            'Y Slice: {}'.format(block_data['object'].find_y_slice()),
            'Possible blocks: {} {}'.format(possible_blocks, y_mult),
            'Block weight: {}'.format(block_data['object'].calculate(debug=2)),
            'Block ID: {}'.format(block_data['object'].calculate())]
            
            
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
