import pygame
import numpy as np


class PaintWindow:

    def __init__(self, width=350, height=500, px_width=7, px_height=10):
        self.width = width
        self.height = height

        self.pxs_w = px_width
        self.pxs_h = px_height

        self.pxl_w = width / px_width
        self.pxl_h = height / px_height
        self.ln_w = 4

        self.white = (255, 255, 255)
        self.gray = (128, 128, 128)
        self.black = (0, 0, 0)
        self.screen = None
        self.draw_on=False
        self.button = 'L'

    def __call__(self):
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.update(pygame.draw.rect(self.screen, self.white, (0, 0, self.width, self.height)))

        self.draw_on = False
        self._draw_grid()

        return self.draw()

    def _draw_grid(self):
        nb_rows = self.pxs_h + 1
        nb_cols = self.pxs_w + 1
        rows = [x * self.pxl_h - (self.ln_w / 2) for x in range(nb_rows)]
        cols = [x * self.pxl_w - (self.ln_w / 2) for x in range(nb_cols)]

        for row in rows:
            ln = pygame.draw.rect(self.screen, self.gray, (0, row, self.width, self.ln_w))
            pygame.display.update(ln)
        for col in cols:
            ln = pygame.draw.rect(self.screen, self.gray, (col, 0, self.ln_w, self.height))
            pygame.display.update(ln)

    def draw(self):
        try:
            while True:
                e = pygame.event.wait()
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_RETURN):
                    raise StopIteration

                if e.type == pygame.MOUSEBUTTONDOWN:
                    #print(e.button, e.pos)
                    self.button = 'R' if e.button == 3 else 'L'
                    self._paint_pixel(e)
                    self.draw_on = True

                if e.type == pygame.MOUSEBUTTONUP:
                    self.draw_on = False

                if e.type == pygame.MOUSEMOTION:
                    if self.draw_on:
                        self._paint_pixel(e)

        except StopIteration:
            pass

        return self._image_to_array()

    def _paint_pixel(self, e):
        rect_x = int(1. * e.pos[0] / self.pxl_w) * self.pxl_w + self.ln_w / 2
        rect_y = int(1. * e.pos[1] / self.pxl_h) * self.pxl_h + self.ln_w / 2
        if self.button == 'R':
            color = self.white
        else:
            color = self.black
        pygame.display.update(pygame.draw.rect(self.screen, color,
                                               (rect_x, rect_y,
                                                self.pxl_w - self.ln_w,
                                                self.pxl_h - self.ln_w)))

    def _image_to_array(self):
        result = np.ones((self.pxs_h, self.pxs_w))

        x_coords = [int((x+0.5) * self.pxl_w) for x in range(self.pxs_w)]
        y_coords = [int((y + 0.5) * self.pxl_h) for y in range(self.pxs_h)]

        for i in range(self.pxs_h):
            for j in range(self.pxs_w):
                x = x_coords[j]
                y = y_coords[i]

                if self.screen.get_at((x,y)) == self.black:
                    result[i,j] = 0

        return result



# uzycie:

#ptw = PaintWindow()
#arr_10_x_7 = ptw()
