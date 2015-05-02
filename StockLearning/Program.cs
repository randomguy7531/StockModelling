using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using StockLearning.FSharp;

namespace StockModelsConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            String dataLocation = @"C:\Users\Stanley\Documents\Visual Studio 2013\Projects\StockLearning\Stock Data\";
            String tickersFile = dataLocation + "Tickers2.txt";

            //read ticker list from the file
            var tickers = DataHelpers.readTickerFile(tickersFile);
            Console.Write(tickers);

            //read data for tickers in list from their files
            var curves = DataHelpers.readStockDataFilesFromTickers(dataLocation, tickers);

            Console.Read();
        }
    }
}
