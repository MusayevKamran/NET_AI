namespace Neural_Network_Learning;

public class NeuralNetwork
{
    private double[,] _weights;

    enum Operation
    {
        Multiply,
        Add,
        Subtract
    }

    public NeuralNetwork()
    {
        Random randomNumber = new Random(1);
        int numberOfInputNodes = 3;
        int numberOfOutputNodes = 1;
        _weights = new double[numberOfInputNodes, numberOfOutputNodes];
        for (int i = 0; i < numberOfInputNodes; i++)
        {
            for (int j = 0; j < numberOfOutputNodes; j++)
            {
                _weights[i, j] = 2 * randomNumber.NextDouble() - 1;
            }
        }
    }

    private double[,] Activate(double[,] matrix, bool isDerivative)
    {
        int numberOfRows = matrix.GetLength(0);
        int numberOfColumns = matrix.GetLength(1);
        double[,] result = new double[numberOfRows, numberOfColumns];
        for (int row = 0; row < numberOfRows; row++)
        {
            for (int column = 0; column < numberOfColumns; column++)
            {
                double sigmoidOutput = result[row, column] = 1 / (1 + Math.Exp(-matrix[row, column]));
                double derivativeSigmoidOutput = result[row, column] = matrix[row, column] * (1 - matrix[row, column]);
                result[row, column] = isDerivative ? derivativeSigmoidOutput : sigmoidOutput;
            }
        }

        return result;
    }

    public void Train(double[,] trainingInputs, double[,] trainingOutputs, int numberOfIterations)
    {
        for (int iterations = 0; iterations < numberOfIterations; iterations++)
        {
            double[,] output = Think(trainingOutputs);
            double[,] error = PerformOperation(trainingOutputs, output, Operation.Subtract);
            double[,] adjustments = DotProduct(Transpose(trainingInputs), PerformOperation(error, Activate(output, true), Operation.Multiply));
            _weights = PerformOperation(_weights, adjustments, Operation.Add);
        }
    }

    private double[,] DotProduct(double[,] matrix1, double[,] matrix2)
    {
        int numberOfRowsInMatrix1 = matrix1.GetLength(0);
        int numberOfColumnsInMatrix1 = matrix1.GetLength(1);

        int numberOfRowsInMatrix2 = matrix2.GetLength(0);
        int numberOfColumnsInMatrix2 = matrix2.GetLength(1);

        double[,] result = new double[numberOfRowsInMatrix1, numberOfColumnsInMatrix2];
        for (int rowInMatrix1 = 0; rowInMatrix1 < numberOfRowsInMatrix1; rowInMatrix1++)
        for (int columnInMatrix2 = 0; columnInMatrix2 < numberOfColumnsInMatrix2; columnInMatrix2++)
        {
            double sum = 0;
            for (int columnInMatrix1 = 0; columnInMatrix1 < numberOfColumnsInMatrix1; columnInMatrix1++)
                sum += matrix1[rowInMatrix1, columnInMatrix1] * matrix2[columnInMatrix1, columnInMatrix2];

            result[rowInMatrix1, columnInMatrix2] = sum;
        }

        return result;
    }

    public double[,] Think(double[,] inputs)
    {
        return Activate(DotProduct(inputs, _weights), false);
    }

    private double[,] PerformOperation(double[,] matrix1, double[,] matrix2, Operation operation)
    {
        int rows = matrix1.GetLength(0);
        int columns = matrix1.GetLength(1);
        double[,] result = new double[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                switch (operation)
                {
                    case Operation.Multiply:
                        result[i, j] = matrix1[i, j] * matrix2[i, j];
                        break;
                    case Operation.Add:
                        result[i, j] = matrix1[i, j] + matrix2[i, j];
                        break;
                    case Operation.Subtract:
                        result[i, j] = matrix1[i, j] - matrix2[i, j];
                        break;
                }
            }
        }

        return result;
    }

    private double[,] Transpose(double[,] matrix)
    {
        return matrix.Cast<double>().ToArray().Transpose(matrix.GetLength(0), matrix.GetLength(1));
    }
}

public static class Extension
{
    public static double[,] Transpose(this double[] array, int rows, int columns)
    {
        double[,] result = new double[columns, rows];
        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                result[column, row] = array[row * columns + column];
            }
        }

        return result;
    }
}

public class NeuralNetworkExecutor{
    public void Run()
    {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        double[,] trainingSetInputs = new double[,] { { 0, 0, 0 }, { 1, 1, 1 }, { 1, 0, 0 } };
        double[,] trainingSetOutputs = new double[,] { { 0 }, { 1 }, { 1 } };
        neuralNetwork.Train(trainingSetInputs, trainingSetOutputs, 1000);
        double[,] output = neuralNetwork.Think(new double[,] { { 0, 1, 0 }, { 0, 0, 0 }, { 0, 0, 1 } });

        PrintMatrix(output);


        static void PrintMatrix(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);
            for (int row = 0; row < rows;row++)
            {
                for (int column = 0; column < columns; column++)
                {
                    Console.Write(Math.Round(matrix[row, column]) + " ");
                }
                Console.WriteLine();
            }
        }
    }
}