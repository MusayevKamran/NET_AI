using TorchSharp;

int[,] maze1 =
{
    //0  1  2  3  4  5  6  7  8  9  10 11
    { 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0 }, //row 0
    { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 }, //row 1
    { 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0 }, //row 2
    { 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0 }, //row 3
    { 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0 }, //row 4
    { 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0 }, //row 5
    { 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0 }, //row 6
    { 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0 }, //row 7
    { 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0 }, //row 8
    { 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0 }, //row 9
    { 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0 }, //row 10
    { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 } //row 11 (start position is (11, 5))
};

const float EPSILON = 0.95f;
const float DISCOUNT_FACTOR = 0.8f;
const float LEARNING_RATE = 0.9f;
const int EPISODES = 1500;
const int START_ROW = 11;
const int START_COLUMN = 5;

MazeSolver solver = new MazeSolver(maze1);
solver.TrainTheModel(EPSILON, DISCOUNT_FACTOR, LEARNING_RATE, EPISODES);
solver.NavigateMaze(START_ROW, START_COLUMN);