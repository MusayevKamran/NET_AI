using Microsoft.ML.Data;

namespace Ml_Net_Learning.Housing;

public class HousingData
{
    [LoadColumn(0)] 
    public float SquareFeet { get; set; }
    [LoadColumn(1)] 
    public float Bedrooms { get; set; }
    [LoadColumn(2)] 
    public float Price { get; set; }
    [LoadColumn(3)] 
    public string Neighborhood { get; set; }
}