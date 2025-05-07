//2D mesh script for ResIPy (run the following in gmsh to generate a triangular mesh with topograpghy)
Mesh.Binary = 0;//specify we want ASCII format
cl=0.12;//define characteristic length
cl_factor= 2;//define characteristic length factor
//Define surface points
Point(1) = {-2.50,0.00,0.00,cl*cl_factor};//topography point
Point(2) = {0.00,0.00,0.00,cl};//electrode
Point(3) = {0.50,0.00,0.00,cl};//electrode
Point(4) = {1.00,0.00,0.00,cl};//electrode
Point(5) = {1.50,0.00,0.00,cl};//electrode
Point(6) = {2.00,0.00,0.00,cl};//electrode
Point(7) = {2.50,0.00,0.00,cl};//electrode
Point(8) = {3.00,0.00,0.00,cl};//electrode
Point(9) = {3.50,0.00,0.00,cl};//electrode
Point(10) = {4.00,0.00,0.00,cl};//electrode
Point(11) = {4.50,0.00,0.00,cl};//electrode
Point(12) = {5.00,0.00,0.00,cl};//electrode
Point(13) = {5.50,0.00,0.00,cl};//electrode
Point(14) = {6.00,0.00,0.00,cl};//electrode
Point(15) = {6.50,0.00,0.00,cl};//electrode
Point(16) = {7.00,0.00,0.00,cl};//electrode
Point(17) = {7.50,0.00,0.00,cl};//electrode
Point(18) = {8.00,0.00,0.00,cl};//electrode
Point(19) = {8.50,0.00,0.00,cl};//electrode
Point(20) = {9.00,0.00,0.00,cl};//electrode
Point(21) = {9.50,0.00,0.00,cl};//electrode
Point(22) = {10.00,0.00,0.00,cl};//electrode
Point(23) = {10.50,0.00,0.00,cl};//electrode
Point(24) = {11.00,0.00,0.00,cl};//electrode
Point(25) = {11.50,0.00,0.00,cl};//electrode
Point(26) = {14.00,0.00,0.00,cl*cl_factor};//topography point
//construct lines between each surface point
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,9};
Line(9) = {9,10};
Line(10) = {10,11};
Line(11) = {11,12};
Line(12) = {12,13};
Line(13) = {13,14};
Line(14) = {14,15};
Line(15) = {15,16};
Line(16) = {16,17};
Line(17) = {17,18};
Line(18) = {18,19};
Line(19) = {19,20};
Line(20) = {20,21};
Line(21) = {21,22};
Line(22) = {22,23};
Line(23) = {23,24};
Line(24) = {24,25};
Line(25) = {25,26};
//add points below surface to make a fine mesh region
Point(27) = {-2.50,0.00,-3.83,cl*cl_factor};//base of smoothed mesh region
Point(28) = {-0.14,0.00,-3.83,cl*cl_factor};//base of smoothed mesh region
Point(29) = {2.21,0.00,-3.83,cl*cl_factor};//base of smoothed mesh region
Point(30) = {4.57,0.00,-3.83,cl*cl_factor};//base of smoothed mesh region
Point(31) = {6.93,0.00,-3.83,cl*cl_factor};//base of smoothed mesh region
Point(32) = {9.29,0.00,-3.83,cl*cl_factor};//base of smoothed mesh region
Point(33) = {11.64,0.00,-3.83,cl*cl_factor};//base of smoothed mesh region
Point(34) = {14.00,0.00,-3.83,cl*cl_factor};//base of smoothed mesh region
//make lines between base of fine mesh region points
Line(26) = {27,28};
Line(27) = {28,29};
Line(28) = {29,30};
Line(29) = {30,31};
Line(30) = {31,32};
Line(31) = {32,33};
Line(32) = {33,34};

//Adding boundaries
//end of boundaries.
//Add lines at leftmost side of fine mesh region.
Line(33) = {1,27};
//Add lines at rightmost side of fine mesh region.
Line(34) = {26,34};
//compile lines into a line loop for a mesh surface/region.
Line Loop(1) = {33, 26, 27, 28, 29, 30, 31, 32, -34, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1};

//Background region (Neumann boundary) points
cln=6.25;//characteristic length for background region
Point(35) = {-60.00,0.00,0.00,cln};//far left upper point
Point(36) = {-60.00,0.00,-38.33,cln};//far left lower point
Point(37) = {71.50,0.00,0.00,cln};//far right upper point
Point(38) = {71.50,0.00,-38.33,cln};//far right lower point
//make lines encompassing all the background points - counter clock wise fashion
Line(35) = {1,35};
Line(36) = {35,36};
Line(37) = {36,38};
Line(38) = {38,37};
Line(39) = {37,26};
//Add line loops and plane surfaces for the Neumann region
Line Loop(2) = {35, 36, 37, 38, 39, 34, -32, -31, -30, -29, -28, -27, -26, -33};
Plane Surface(1) = {1, 2};//Coarse mesh region surface

//Adding polygons
//end of polygons.
Plane Surface(2) = {1};//Fine mesh region surface

//Make a physical surface
Physical Surface(1) = {2, 1};

//End gmsh script
