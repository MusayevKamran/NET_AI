namespace TorchSharp;

public class MazeSolver
{
    private readonly int[,] _maze;
    private int[,]? _rewards;
    private torch.Tensor? _qValues;
    private readonly string[] _actions = { "up", "down", "left", "right" };

    private const int WALL_REWARD_VALUE = -500;
    private const int FLOOR_REWARD_VALUE = -10;
    private const int GOAL_REWARD_VALUE = 500;

    public MazeSolver(int[,] maze)
    {
        _maze = maze;
        SetupRewards();
        SetupQValues();
    }

    private void SetupRewards()
    {
        int mazeRows = _maze.GetLength(0);
        int mazeColumns = _maze.GetLength(1);

        _rewards = new int[mazeRows, mazeColumns];

        for (int i = 0; i < mazeRows; i++)
        {
            for (int j = 0; j < mazeColumns; j++)
            {
                switch (_maze[i, j])
                {
                    case 0:
                        _rewards[i, j] = WALL_REWARD_VALUE;
                        break;
                    case 1:
                        _rewards[i, j] = FLOOR_REWARD_VALUE;
                        break;
                    case 2:
                        _rewards[i, j] = GOAL_REWARD_VALUE;
                        break;
                }
            }
        }
    }

    private void SetupQValues()
    {
        int mazeRows = _maze.GetLength(0);
        int mazeColumns = _maze.GetLength(1);

        _qValues = torch.zeros(mazeRows, mazeColumns, _actions.Length);
    }

    private bool HasHitWallOrEndOfMaze(int currentRow, int currentColumn)
    {
        return _rewards[currentRow, currentColumn] != FLOOR_REWARD_VALUE;
    }

    private long DetermineNextAction(int currentRow, int currentColumn, float epsilon)
    {
        Random random = new Random();
        double randomBeetwen0And1 = random.NextDouble();
        long nextAction = randomBeetwen0And1 < epsilon ? torch.argmax(_qValues[currentRow, currentColumn]).item<long>() : random.Next(_actions.Length);
        return nextAction;
    }

    private (int, int) MoveOneSpace(int currentRow, int currentColumn, long currentAction)
    {
        int mazeRows = _maze.GetLength(0);
        int mazeColumns = _maze.GetLength(1);

        int nextRow = currentRow;
        int nextColumn = currentColumn;

        if (_actions[currentAction] == "up" && currentRow > 0)
        {
            nextRow--;
        }
        else if (_actions[currentAction] == "down" && currentRow < mazeRows - 1)
        {
            nextRow++;
        }
        else if (_actions[currentAction] == "left" && currentColumn > 0)
        {
            nextColumn--;
        }
        else if (_actions[currentAction] == "right" && currentColumn < mazeColumns - 1)
        {
            nextColumn++;
        }

        return (nextRow, nextColumn);
    }

    public void TrainTheModel(float epsilon, float discountFactor, float learningRate, int episodes)
    {
        for (int episode = 0; episode < episodes; episode++)
        {
            Console.WriteLine("--------Starting episode " + episode + "--------");
            int currentRow = 11;
            int currentColumn = 5;
            while (!HasHitWallOrEndOfMaze(currentRow, currentColumn))
            {
                long currentAction = DetermineNextAction(currentRow, currentColumn, epsilon);
                int previousRow = currentRow;
                int previousColumn = currentColumn;
                (int, int) nextMove = MoveOneSpace(currentRow, currentColumn, currentAction);
                currentRow = nextMove.Item1;
                currentColumn = nextMove.Item2;
                float reward = _rewards[currentRow, currentColumn];
                float previousQValue = _qValues[previousRow, previousColumn, currentAction].item<float>();
                float temporalDifference = reward + (discountFactor * torch.max(_qValues[currentRow, currentColumn]).item<float>()) - previousQValue;
                float newQValue = previousQValue + (learningRate * temporalDifference);
                _qValues[previousRow, previousColumn, currentAction] = newQValue;
            }

            Console.WriteLine("--------Finished episode " + episode + "--------");
        }

        Console.WriteLine("Completed training");
    }

    public List<int[]> NavigateMaze(int startRow, int startColumn)
    {
        List<int[]> path = new List<int[]>();
        if (HasHitWallOrEndOfMaze(startRow, startColumn))
        {
            return new List<int[]>();
        }

        int currentRow = startRow;
        int currentColumn = startColumn;
        path.Add(new int[] { currentRow, currentColumn });

        while (!HasHitWallOrEndOfMaze(currentRow, currentColumn))
        {
            int nextAction = (int)DetermineNextAction(currentRow, currentColumn, 1.0f);
            (int, int) nextMove = MoveOneSpace(currentRow, currentColumn, nextAction);
            currentRow = nextMove.Item1;
            currentColumn = nextMove.Item2;
            if (_rewards[currentRow, currentColumn] != WALL_REWARD_VALUE)
            {
                path.Add(new int[] { currentRow, currentColumn });
            }
        }

        int moveCount = 1;
        for (int i = 0; i < path.Count; i++)
        {
            Console.Write("Move " + moveCount + ": (");
            foreach (int element in path[i])
            {
                Console.Write(" " + element);
            }

            Console.Write(")");
            Console.WriteLine();
            moveCount++;
        }

        return path;
    }
}

public  class MazeSolverExecutor
{
    public void Run()
    {
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
    }
}