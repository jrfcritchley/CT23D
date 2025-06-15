import cv2
import os
import numpy as np
#http://paulbourke.net/geometry/polygonise/            TRIANGULATION TABLE

#threshold for determining surface by pixel brightness
surfLevel = 100


#Edge lookup table
edgeTable=[
0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0]

#Triangulation lookup table
triTable=[
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 3, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 3, 1, 8, 1, 9,-1,-1,-1,-1,-1,-1,-1],
    [10, 1, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 3, 0, 1, 2,10,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 2, 9, 2,10,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 2, 8, 2,10, 8, 8,10, 9,-1,-1,-1,-1],
    [11, 2, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 2, 0,11, 0, 8,-1,-1,-1,-1,-1,-1,-1],
    [11, 2, 3, 0, 1, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 1,11, 1, 9,11,11, 9, 8,-1,-1,-1,-1],
    [10, 1, 3,10, 3,11,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 0,10, 0, 8,10,10, 8,11,-1,-1,-1,-1],
    [ 0, 3, 9, 3,11, 9, 9,11,10,-1,-1,-1,-1],
    [ 8,10, 9, 8,11,10,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 4, 3, 4, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 9, 0, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 4, 1, 4, 7, 1, 1, 7, 3,-1,-1,-1,-1],
    [10, 1, 2, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 2,10, 1, 0, 4, 7, 0, 7, 3,-1,-1,-1,-1],
    [ 4, 7, 8, 0, 2,10, 0,10, 9,-1,-1,-1,-1],
    [ 2, 7, 3, 2, 9, 7, 7, 9, 4, 2,10, 9,-1],
    [ 2, 3,11, 7, 8, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 7,11, 4,11, 2, 4, 4, 2, 0,-1,-1,-1,-1],
    [ 3,11, 2, 4, 7, 8, 9, 0, 1,-1,-1,-1,-1],
    [ 2, 7,11, 2, 1, 7, 1, 4, 7, 1, 9, 4,-1],
    [ 8, 4, 7,11,10, 1,11, 1, 3,-1,-1,-1,-1],
    [11, 4, 7, 1, 4,11, 1,11,10, 1, 0, 4,-1],
    [ 3, 8, 0, 7,11, 4,11, 9, 4,11,10, 9,-1],
    [ 7,11, 4, 4,11, 9,11,10, 9,-1,-1,-1,-1],
    [ 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 8, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 4, 0, 5, 0, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 8, 5, 8, 3, 5, 5, 3, 1,-1,-1,-1,-1],
    [ 2,10, 1, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 5, 4, 9,10, 1, 2,-1,-1,-1,-1],
    [10, 5, 2, 5, 4, 2, 2, 4, 0,-1,-1,-1,-1],
    [ 3, 4, 8, 3, 2, 4, 2, 5, 4, 2,10, 5,-1],
    [11, 2, 3, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 4, 8,11, 2, 8, 2, 0,-1,-1,-1,-1],
    [ 3,11, 2, 1, 5, 4, 1, 4, 0,-1,-1,-1,-1],
    [ 8, 5, 4, 2, 5, 8, 2, 8,11, 2, 1, 5,-1],
    [ 5, 4, 9, 1, 3,11, 1,11,10,-1,-1,-1,-1],
    [ 0, 9, 1, 4, 8, 5, 8,10, 5, 8,11,10,-1],
    [ 3, 4, 0, 3,10, 4, 4,10, 5, 3,11,10,-1],
    [ 4, 8, 5, 5, 8,10, 8,11,10,-1,-1,-1,-1],
    [ 9, 5, 7, 9, 7, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 9, 3, 9, 5, 3, 3, 5, 7,-1,-1,-1,-1],
    [ 8, 0, 7, 0, 1, 7, 7, 1, 5,-1,-1,-1,-1],
    [ 1, 7, 3, 1, 5, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 5, 7, 8, 5, 8, 9,-1,-1,-1,-1],
    [ 9, 1, 0,10, 5, 2, 5, 3, 2, 5, 7, 3,-1],
    [ 5, 2,10, 8, 2, 5, 8, 5, 7, 8, 0, 2,-1],
    [10, 5, 2, 2, 5, 3, 5, 7, 3,-1,-1,-1,-1],
    [11, 2, 3, 8, 9, 5, 8, 5, 7,-1,-1,-1,-1],
    [ 9, 2, 0, 9, 7, 2, 2, 7,11, 9, 5, 7,-1],
    [ 0, 3, 8, 2, 1,11, 1, 7,11, 1, 5, 7,-1],
    [ 2, 1,11,11, 1, 7, 1, 5, 7,-1,-1,-1,-1],
    [ 3, 9, 1, 3, 8, 9, 7,11,10, 7,10, 5,-1],
    [ 9, 1, 0,10, 7,11,10, 5, 7,-1,-1,-1,-1],
    [ 3, 8, 0, 7,10, 5, 7,11,10,-1,-1,-1,-1],
    [11, 5, 7,11,10, 5,-1,-1,-1,-1,-1,-1,-1],
    [10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 3, 0,10, 6, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9, 5,10, 6,-1,-1,-1,-1,-1,-1,-1],
    [10, 6, 5, 9, 8, 3, 9, 3, 1,-1,-1,-1,-1],
    [ 1, 2, 6, 1, 6, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 2, 6, 5, 2, 5, 1,-1,-1,-1,-1],
    [ 5, 9, 6, 9, 0, 6, 6, 0, 2,-1,-1,-1,-1],
    [ 9, 6, 5, 3, 6, 9, 3, 9, 8, 3, 2, 6,-1],
    [ 3,11, 2,10, 6, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 6, 5,10, 2, 0, 8, 2, 8,11,-1,-1,-1,-1],
    [ 1, 9, 0, 6, 5,10,11, 2, 3,-1,-1,-1,-1],
    [ 1,10, 2, 5, 9, 6, 9,11, 6, 9, 8,11,-1],
    [11, 6, 3, 6, 5, 3, 3, 5, 1,-1,-1,-1,-1],
    [ 0, 5, 1, 0,11, 5, 5,11, 6, 0, 8,11,-1],
    [ 0, 5, 9, 0, 3, 5, 3, 6, 5, 3,11, 6,-1],
    [ 5, 9, 6, 6, 9,11, 9, 8,11,-1,-1,-1,-1],
    [10, 6, 5, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 5,10, 6, 7, 3, 0, 7, 0, 4,-1,-1,-1,-1],
    [ 5,10, 6, 0, 1, 9, 8, 4, 7,-1,-1,-1,-1],
    [ 4, 5, 9, 6, 7,10, 7, 1,10, 7, 3, 1,-1],
    [ 7, 8, 4, 5, 1, 2, 5, 2, 6,-1,-1,-1,-1],
    [ 4, 1, 0, 4, 5, 1, 6, 7, 3, 6, 3, 2,-1],
    [ 9, 4, 5, 8, 0, 7, 0, 6, 7, 0, 2, 6,-1],
    [ 4, 5, 9, 6, 3, 2, 6, 7, 3,-1,-1,-1,-1],
    [ 7, 8, 4, 2, 3,11,10, 6, 5,-1,-1,-1,-1],
    [11, 6, 7,10, 2, 5, 2, 4, 5, 2, 0, 4,-1],
    [11, 6, 7, 8, 0, 3, 1,10, 2, 9, 4, 5,-1],
    [ 6, 7,11, 1,10, 2, 9, 4, 5,-1,-1,-1,-1],
    [ 6, 7,11, 4, 5, 8, 5, 3, 8, 5, 1, 3,-1],
    [ 6, 7,11, 4, 1, 0, 4, 5, 1,-1,-1,-1,-1],
    [ 4, 5, 9, 3, 8, 0,11, 6, 7,-1,-1,-1,-1],
    [ 9, 4, 5, 7,11, 6,-1,-1,-1,-1,-1,-1,-1],
    [10, 6, 4,10, 4, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 3, 0, 9,10, 6, 9, 6, 4,-1,-1,-1,-1],
    [ 1,10, 0,10, 6, 0, 0, 6, 4,-1,-1,-1,-1],
    [ 8, 6, 4, 8, 1, 6, 6, 1,10, 8, 3, 1,-1],
    [ 9, 1, 4, 1, 2, 4, 4, 2, 6,-1,-1,-1,-1],
    [ 1, 0, 9, 3, 2, 8, 2, 4, 8, 2, 6, 4,-1],
    [ 2, 4, 0, 2, 6, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 2, 8, 8, 2, 4, 2, 6, 4,-1,-1,-1,-1],
    [ 2, 3,11, 6, 4, 9, 6, 9,10,-1,-1,-1,-1],
    [ 0,10, 2, 0, 9,10, 4, 8,11, 4,11, 6,-1],
    [10, 2, 1,11, 6, 3, 6, 0, 3, 6, 4, 0,-1],
    [10, 2, 1,11, 4, 8,11, 6, 4,-1,-1,-1,-1],
    [ 1, 4, 9,11, 4, 1,11, 1, 3,11, 6, 4,-1],
    [ 0, 9, 1, 4,11, 6, 4, 8,11,-1,-1,-1,-1],
    [11, 6, 3, 3, 6, 0, 6, 4, 0,-1,-1,-1,-1],
    [ 8, 6, 4, 8,11, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 6, 7,10, 7, 8,10,10, 8, 9,-1,-1,-1,-1],
    [ 9, 3, 0, 6, 3, 9, 6, 9,10, 6, 7, 3,-1],
    [ 6, 1,10, 6, 7, 1, 7, 0, 1, 7, 8, 0,-1],
    [ 6, 7,10,10, 7, 1, 7, 3, 1,-1,-1,-1,-1],
    [ 7, 2, 6, 7, 9, 2, 2, 9, 1, 7, 8, 9,-1],
    [ 1, 0, 9, 3, 6, 7, 3, 2, 6,-1,-1,-1,-1],
    [ 8, 0, 7, 7, 0, 6, 0, 2, 6,-1,-1,-1,-1],
    [ 2, 7, 3, 2, 6, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 7,11, 6, 3, 8, 2, 8,10, 2, 8, 9,10,-1],
    [11, 6, 7,10, 0, 9,10, 2, 0,-1,-1,-1,-1],
    [ 2, 1,10, 7,11, 6, 8, 0, 3,-1,-1,-1,-1],
    [ 1,10, 2, 6, 7,11,-1,-1,-1,-1,-1,-1,-1],
    [ 7,11, 6, 3, 9, 1, 3, 8, 9,-1,-1,-1,-1],
    [ 9, 1, 0,11, 6, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 3, 8,11, 6, 7,-1,-1,-1,-1,-1,-1,-1],
    [11, 6, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3,11, 7, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 1,11, 7, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 6,11, 3, 1, 9, 3, 9, 8,-1,-1,-1,-1],
    [ 1, 2,10, 6,11, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 2,10, 1, 7, 6,11, 8, 3, 0,-1,-1,-1,-1],
    [11, 7, 6,10, 9, 0,10, 0, 2,-1,-1,-1,-1],
    [ 7, 6,11, 3, 2, 8, 8, 2,10, 8,10, 9,-1],
    [ 2, 3, 7, 2, 7, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 7, 0, 7, 6, 0, 0, 6, 2,-1,-1,-1,-1],
    [ 1, 9, 0, 3, 7, 6, 3, 6, 2,-1,-1,-1,-1],
    [ 7, 6, 2, 7, 2, 9, 2, 1, 9, 7, 9, 8,-1],
    [ 6,10, 7,10, 1, 7, 7, 1, 3,-1,-1,-1,-1],
    [ 6,10, 1, 6, 1, 7, 7, 1, 0, 7, 0, 8,-1],
    [ 9, 0, 3, 6, 9, 3, 6,10, 9, 6, 3, 7,-1],
    [ 6,10, 7, 7,10, 8,10, 9, 8,-1,-1,-1,-1],
    [ 8, 4, 6, 8, 6,11,-1,-1,-1,-1,-1,-1,-1],
    [11, 3, 6, 3, 0, 6, 6, 0, 4,-1,-1,-1,-1],
    [ 0, 1, 9, 4, 6,11, 4,11, 8,-1,-1,-1,-1],
    [ 1, 9, 4,11, 1, 4,11, 3, 1,11, 4, 6,-1],
    [10, 1, 2,11, 8, 4,11, 4, 6,-1,-1,-1,-1],
    [10, 1, 2,11, 3, 6, 6, 3, 0, 6, 0, 4,-1],
    [ 0, 2,10, 0,10, 9, 4,11, 8, 4, 6,11,-1],
    [ 2,11, 3, 6, 9, 4, 6,10, 9,-1,-1,-1,-1],
    [ 3, 8, 2, 8, 4, 2, 2, 4, 6,-1,-1,-1,-1],
    [ 2, 0, 4, 2, 4, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 9, 0, 3, 8, 2, 2, 8, 4, 2, 4, 6,-1],
    [ 9, 4, 1, 1, 4, 2, 4, 6, 2,-1,-1,-1,-1],
    [ 8, 4, 6, 8, 6, 1, 6,10, 1, 8, 1, 3,-1],
    [ 1, 0,10,10, 0, 6, 0, 4, 6,-1,-1,-1,-1],
    [ 8, 0, 3, 9, 6,10, 9, 4, 6,-1,-1,-1,-1],
    [10, 4, 6,10, 9, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 4, 7, 6,11,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 5, 3, 0, 8,11, 7, 6,-1,-1,-1,-1],
    [ 6,11, 7, 4, 0, 1, 4, 1, 5,-1,-1,-1,-1],
    [ 6,11, 7, 4, 8, 5, 5, 8, 3, 5, 3, 1,-1],
    [ 6,11, 7, 1, 2,10, 9, 5, 4,-1,-1,-1,-1],
    [11, 7, 6, 8, 3, 0, 1, 2,10, 9, 5, 4,-1],
    [11, 7, 6,10, 5, 2, 2, 5, 4, 2, 4, 0,-1],
    [ 7, 4, 8, 2,11, 3,10, 5, 6,-1,-1,-1,-1],
    [ 4, 9, 5, 6, 2, 3, 6, 3, 7,-1,-1,-1,-1],
    [ 9, 5, 4, 8, 7, 0, 0, 7, 6, 0, 6, 2,-1],
    [ 4, 0, 1, 4, 1, 5, 6, 3, 7, 6, 2, 3,-1],
    [ 7, 4, 8, 5, 2, 1, 5, 6, 2,-1,-1,-1,-1],
    [ 4, 9, 5, 6,10, 7, 7,10, 1, 7, 1, 3,-1],
    [ 5, 6,10, 0, 9, 1, 8, 7, 4,-1,-1,-1,-1],
    [ 5, 6,10, 7, 0, 3, 7, 4, 0,-1,-1,-1,-1],
    [10, 5, 6, 4, 8, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 6, 9, 6,11, 9, 9,11, 8,-1,-1,-1,-1],
    [ 0, 9, 5, 0, 5, 3, 3, 5, 6, 3, 6,11,-1],
    [ 0, 1, 5, 0, 5,11, 5, 6,11, 0,11, 8,-1],
    [11, 3, 6, 6, 3, 5, 3, 1, 5,-1,-1,-1,-1],
    [ 1, 2,10, 5, 6, 9, 9, 6,11, 9,11, 8,-1],
    [ 1, 0, 9, 6,10, 5,11, 3, 2,-1,-1,-1,-1],
    [ 6,10, 5, 2, 8, 0, 2,11, 8,-1,-1,-1,-1],
    [ 3, 2,11,10, 5, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 6, 3, 9, 6, 3, 8, 9, 3, 6, 2,-1],
    [ 5, 6, 9, 9, 6, 0, 6, 2, 0,-1,-1,-1,-1],
    [ 0, 3, 8, 2, 5, 6, 2, 1, 5,-1,-1,-1,-1],
    [ 1, 6, 2, 1, 5, 6,-1,-1,-1,-1,-1,-1,-1],
    [10, 5, 6, 9, 3, 8, 9, 1, 3,-1,-1,-1,-1],
    [ 0, 9, 1, 5, 6,10,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 0, 3,10, 5, 6,-1,-1,-1,-1,-1,-1,-1],
    [10, 5, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 7, 5,11, 5,10,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 8, 7, 5,10, 7,10,11,-1,-1,-1,-1],
    [ 9, 0, 1,10,11, 7,10, 7, 5,-1,-1,-1,-1],
    [ 3, 1, 9, 3, 9, 8, 7,10,11, 7, 5,10,-1],
    [ 2,11, 1,11, 7, 1, 1, 7, 5,-1,-1,-1,-1],
    [ 0, 8, 3, 2,11, 1, 1,11, 7, 1, 7, 5,-1],
    [ 9, 0, 2, 9, 2, 7, 2,11, 7, 9, 7, 5,-1],
    [11, 3, 2, 8, 5, 9, 8, 7, 5,-1,-1,-1,-1],
    [10, 2, 5, 2, 3, 5, 5, 3, 7,-1,-1,-1,-1],
    [ 5,10, 2, 8, 5, 2, 8, 7, 5, 8, 2, 0,-1],
    [ 9, 0, 1,10, 2, 5, 5, 2, 3, 5, 3, 7,-1],
    [ 1,10, 2, 5, 8, 7, 5, 9, 8,-1,-1,-1,-1],
    [ 1, 3, 7, 1, 7, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 7, 0, 0, 7, 1, 7, 5, 1,-1,-1,-1,-1],
    [ 0, 3, 9, 9, 3, 5, 3, 7, 5,-1,-1,-1,-1],
    [ 9, 7, 5, 9, 8, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 5, 8, 5,10, 8, 8,10,11,-1,-1,-1,-1],
    [ 3, 0, 4, 3, 4,10, 4, 5,10, 3,10,11,-1],
    [ 0, 1, 9, 4, 5, 8, 8, 5,10, 8,10,11,-1],
    [ 5, 9, 4, 1,11, 3, 1,10,11,-1,-1,-1,-1],
    [ 8, 4, 5, 2, 8, 5, 2,11, 8, 2, 5, 1,-1],
    [ 3, 2,11, 1, 4, 5, 1, 0, 4,-1,-1,-1,-1],
    [ 9, 4, 5, 8, 2,11, 8, 0, 2,-1,-1,-1,-1],
    [11, 3, 2, 9, 4, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 8, 4, 3, 4, 2, 2, 4, 5, 2, 5,10,-1],
    [10, 2, 5, 5, 2, 4, 2, 0, 4,-1,-1,-1,-1],
    [ 0, 3, 8, 5, 9, 4,10, 2, 1,-1,-1,-1,-1],
    [ 2, 1,10, 9, 4, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 5, 8, 8, 5, 3, 5, 1, 3,-1,-1,-1,-1],
    [ 5, 0, 4, 5, 1, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 8, 0, 4, 5, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 4, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 4,11, 4, 9,11,11, 9,10,-1,-1,-1,-1],
    [ 3, 0, 8, 7, 4,11,11, 4, 9,11, 9,10,-1],
    [11, 7, 4, 1,11, 4, 1,10,11, 1, 4, 0,-1],
    [ 8, 7, 4,11, 1,10,11, 3, 1,-1,-1,-1,-1],
    [ 2,11, 7, 2, 7, 1, 1, 7, 4, 1, 4, 9,-1],
    [ 3, 2,11, 4, 8, 7, 9, 1, 0,-1,-1,-1,-1],
    [ 7, 4,11,11, 4, 2, 4, 0, 2,-1,-1,-1,-1],
    [ 2,11, 3, 7, 4, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 3, 7, 2, 7, 9, 7, 4, 9, 2, 9,10,-1],
    [ 4, 8, 7, 0,10, 2, 0, 9,10,-1,-1,-1,-1],
    [ 2, 1,10, 0, 7, 4, 0, 3, 7,-1,-1,-1,-1],
    [10, 2, 1, 8, 7, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 1, 4, 4, 1, 7, 1, 3, 7,-1,-1,-1,-1],
    [ 1, 0, 9, 8, 7, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 4, 0, 3, 7, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 7, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 9,10, 8,10,11,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 9, 3, 3, 9,11, 9,10,11,-1,-1,-1,-1],
    [ 1,10, 0, 0,10, 8,10,11, 8,-1,-1,-1,-1],
    [10, 3, 1,10,11, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 2,11, 1, 1,11, 9,11, 8, 9,-1,-1,-1,-1],
    [11, 3, 2, 0, 9, 1,-1,-1,-1,-1,-1,-1,-1],
    [11, 0, 2,11, 8, 0,-1,-1,-1,-1,-1,-1,-1],
    [11, 3, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 8, 2, 2, 8,10, 8, 9,10,-1,-1,-1,-1],
    [ 9, 2, 0, 9,10, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 0, 3, 1,10, 2,-1,-1,-1,-1,-1,-1,-1],
    [10, 2, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 1, 3, 8, 9, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 1, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 0, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
];

#take images from a file and put them in the order taken from the file, the function outputs an array of the images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return(images)

#------------------------------------------------------------------------------------------------------------------------------#
#this can be done by creating an array of the vertices and using the filter or map functions
#it can also be done with generators via constucting a string and adding the 0's and 1's iterativley
#I dont know if either option is faster but im going to guess that 8 arithmatic operations are faster 
#This should be first to go if its slow though
#------------------------------------------------------------------------------------------------------------------------------#




def bordCheck(vert0,vert1,vert2,vert3,vert4,vert5,vert6,vert7):
    byteTotal = 0
    
    if(vert0>surfLevel):
        byteTotal +=1
    if(vert1>surfLevel):
        byteTotal +=2
    if(vert2>surfLevel):
        byteTotal +=4
    if(vert3>surfLevel):
        byteTotal +=8
    if(vert4>surfLevel):
        byteTotal +=16
    if(vert5>surfLevel):
        byteTotal +=32
    if(vert6>surfLevel):
        byteTotal +=64
    if(vert7>surfLevel):
        byteTotal +=128
    
    return(byteTotal)



def Lookups(cubeValue):

    #this finds the hex value stored in the edge table, converts it to a string so it can be converted to an interger and then to binary
    # the 2: slices it as an array to remove the "0b" that python adds to all binary numbers at the start
    binVal = bin(edgeTable[cubeValue])[2:]

    #this formats the string into a uniform 12 bits as python automatically removes the leading 0's the str-list conversions are slow
    #unfortunatley i dont know a faster way and couldnt find a faster way
    binList = list(binVal)
    binList.insert(0,(12 - len(binVal))*"0")
    binTwelve = ''.join(binList)
    return(binTwelve)


def edgePos(v1pos,v1val,v2pos,v2val):

    if((surfLevel-v1val) == 0):
        return(v1pos)
    elif((surfLevel-v2val) == 0):
        return(v2pos)
    else:

    
    #this is to linearly interpolate between the inside and outside vertices to find where the vertice should be placed on the cut edge
    #this only works if v1val is SMALLER than the isovalue/surfLevel, if everything is a mess this is why.
        intersectPos = v1pos + ((surfLevel - v1val)*(v2pos-v1pos))/(v2val-v1val)
        
        return(intersectPos)

def pointSwap(pos1,val1,pos2,val2):
    if(val2 < val1):
        tempVal = val1
        tempPos = pos1
        val1 = val2
        pos1 = pos2
        val2 = tempVal
        pos2 = tempPos
    return(pos1,val1,pos2,val2)


#creates a list containing ascending values of intergers in the same order of a list with other intergers in
#eg [3, 10, 1, 11, 10, 3] becomes [1, 2, 0, 3, 2, 1]
def ascOrder(lst):
    sorted_lst = sorted(set(lst))
    return [sorted_lst.index(bf) for bf in lst]


#divides an array into an array containing arrays which are 3 elements in size
def divArray(arr):
    return [arr[i:i+3] for i in range(0, len(arr), 3)]



def calculate_triangle_normal(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    # Convert the input values to numpy arrays
    v1 = np.array([x1, y1, z1])
    v2 = np.array([x2, y2, z2])
    v3 = np.array([x3, y3, z3])

    # Calculate two vectors that lie on the triangle's surface
    vector1 = v2 - v1
    vector2 = v3 - v1

    # Calculate the cross product of the two vectors to get the normal vector
    normal = np.cross(vector1, vector2)

    # Normalize the normal vector
    if np.all(normal == 0):
        normal = np.array([0, 0, 0])
    else:
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)

    # Convert the normal vector to a string in the format required by the OBJ file
    normal_string = f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n"

    return(normal_string)
            
#loads images into array
scans = load_images_from_folder("D:/Users/Jack Critchley/Desktop/MarchingCubes")


#count for entries into the obj file, 
entryCount=1
trentCount = 1
#opens the obj file, using the with statement to ensure its closed in the event of an error
with open("D:/Users/Jack Critchley/Desktop/MarchingCubes/3DModel.obj", "w") as file:


    #gets the number of rows and columns 
    height = scans[0].shape[0]
    width = scans[0].shape[1]

    #------------------------------------------------------------------------------------------------------------------------------#
    #potential to accelerate iteration step by removing large chunks of dead space via divide and conquer
    #essentialy create a list of pixels to check through by removing areas with nothing in
    #the list of pixels with nothing in is run once and just checks if a large area has anything in
    #this recurses by some constant, probably also a cool algo or math thing to find constant
    #this is called quadtrees
    #------------------------------------------------------------------------------------------------------------------------------#
    for i in range(len(scans)):
        scans[i] = cv2.cvtColor(scans[i], cv2.COLOR_BGR2GRAY).astype(int)


    #iterate through scans
    for i in range((len(scans)-1)):
        print(f"Currently on Scan {i}")
        #iterate through the image pixel by pixel
        height = scans[0].shape[0]
        width = scans[0].shape[1]
        
        for y in range(0,(height-3)):
            for x in range(0,(width-3)):

                #creates the voxel
                v0val = scans[i][y,x]
                v1val = scans[i][y,x+1]
                v2val = scans[i][y+1,x+1]
                v3val = scans[i][y+1,x]
                v4val = scans[i+1][y,x]
                v5val = scans[i+1][y,x+1]
                v6val = scans[i+1][y+1,x+1]
                v7val = scans[i+1][y+1,x]

                
                #instatiates/clears the edge array
                cutEdge=[]

                #finds the cube index (8 bits)
                cubeIndex = bordCheck(v0val,v1val,v2val,v3val,v4val,v5val,v6val,v7val)

                #this just ensures time is not wasted on completley empty space
                if(cubeIndex == 0 or cubeIndex == 255):
                    continue
                else:
            
                    #finds the edges which are intersected(12 bits)
                    edgeIndex = Lookups(cubeIndex)

                    #this finds which edges have been cut, finds the according vertices and finds the interpolated value between them
                    #it outputs the position if the vertice and the corresponding edge name in a array
                    
                    for j in range(len(edgeIndex)):
                        if (j == 0 and edgeIndex[::-1][j] == "1"):
                            sanitPoints = pointSwap(x,v0val,x+1,v1val)
                            cutEdge.append([edgePos(sanitPoints[0],sanitPoints[1],sanitPoints[2],sanitPoints[3]),y,i,j])
                        if (j == 1 and edgeIndex[::-1][j] == "1"):
                            sanitPoints = pointSwap(y,v1val,y+1,v2val)
                            cutEdge.append([x+1,edgePos(sanitPoints[0],sanitPoints[1],sanitPoints[2],sanitPoints[3]),i,j])
                        if (j == 2 and edgeIndex[::-1][j] == "1"):
                            sanitPoints = pointSwap(x+1,v2val,x,v3val)
                            cutEdge.append([edgePos(sanitPoints[0],sanitPoints[1],sanitPoints[2],sanitPoints[3]),y+1,i,j])
                        if (j == 3 and edgeIndex[::-1][j] == "1"):
                            sanitPoints = pointSwap(y+1,v3val,y,v0val)
                            cutEdge.append([x,edgePos(sanitPoints[0],sanitPoints[1],sanitPoints[2],sanitPoints[3]),i,j])
                        if (j == 4 and edgeIndex[::-1][j] == "1"):
                            sanitPoints = pointSwap(x,v4val,x+1,v5val)
                            cutEdge.append([edgePos(sanitPoints[0],sanitPoints[1],sanitPoints[2],sanitPoints[3]),y,i+1,j])
                        if (j == 5 and edgeIndex[::-1][j] == "1"):
                            sanitPoints = pointSwap(y,v5val,y+1,v6val)
                            cutEdge.append([x+1,edgePos(sanitPoints[0],sanitPoints[1],sanitPoints[2],sanitPoints[3]),i+1,j])
                        if (j == 6 and edgeIndex[::-1][j] == "1"):
                            sanitPoints = pointSwap(x+1,v6val,x,v7val)
                            cutEdge.append([edgePos(sanitPoints[0],sanitPoints[1],sanitPoints[2],sanitPoints[3]),y+1,i+1,j])
                        if (j == 7 and edgeIndex[::-1][j] == "1"):
                            sanitPoints = pointSwap(y+1,v7val,y,v4val)
                            cutEdge.append([x,edgePos(sanitPoints[0],sanitPoints[1],sanitPoints[2],sanitPoints[3]),i+1,j])
                        if (j == 8 and edgeIndex[::-1][j] == "1"):
                            sanitPoints = pointSwap(i,v0val,i+1,v4val)
                            cutEdge.append([x,y,edgePos(sanitPoints[0],sanitPoints[1],sanitPoints[2],sanitPoints[3]),j])
                        if (j == 9 and edgeIndex[::-1][j] == "1"):
                            sanitPoints = pointSwap(i,v1val,i+1,v5val)
                            cutEdge.append([x+1,y,edgePos(sanitPoints[0],sanitPoints[1],sanitPoints[2],sanitPoints[3]),j])
                        if (j == 10 and edgeIndex[::-1][j] == "1"):
                            sanitPoints = pointSwap(i,v2val,i+1,v6val)
                            cutEdge.append([x+1,y+1,edgePos(sanitPoints[0],sanitPoints[1],sanitPoints[2],sanitPoints[3]),j])
                        if (j == 11 and edgeIndex[::-1][j] == "1"):
                            sanitPoints = pointSwap(i,v3val,i+1,v7val)
                            cutEdge.append([x,y+1,edgePos(sanitPoints[0],sanitPoints[1],sanitPoints[2],sanitPoints[3]),j])

                    #finds the array in triangulation table which informs the order of vertex connection
                    facetSet = triTable[cubeIndex]

                    #removes the -1's
                    pointSet = list(filter(lambda cv: cv!=-1, facetSet))
        
                    rordSet = divArray(ascOrder(pointSet))

                    
##                    print(cubeIndex)
##                    print(edgeIndex)
##                    print(cutEdge)
##                    print(facetSet)
##                    print(rordSet)
##                    print(len(rordSet))
                    
                    for k in range(len(cutEdge)):
                        file.write('v {} {} {}\n'.format(round(cutEdge[k][0],6), round(cutEdge[k][1],6), round(cutEdge[k][2],6)))

                    for l in range(len(rordSet)):
                        normVal = calculate_triangle_normal(cutEdge[rordSet[l][0]][0], cutEdge[rordSet[l][0]][1], cutEdge[rordSet[l][0]][2],cutEdge[rordSet[l][1]][0], cutEdge[rordSet[l][1]][1], cutEdge[rordSet[l][1]][2],cutEdge[rordSet[l][2]][0], cutEdge[rordSet[l][2]][1], cutEdge[rordSet[l][2]][2])
                        file.write(normVal)
                    
                    for r in range(len(rordSet)):
                        #file.write('f {} {} {}\n'.format(rordSet[r][0] + entryCount, rordSet[r][1] + entryCount,rordSet[r][2] + entryCount))
                        file.write('f {}//{} {}//{} {}//{}\n'.format(rordSet[r][0] + entryCount,trentCount, rordSet[r][1] + entryCount,trentCount,rordSet[r][2] + entryCount,trentCount))
                        trentCount =trentCount+1
                    entryCount = entryCount + len(cutEdge)
                
file.close()


