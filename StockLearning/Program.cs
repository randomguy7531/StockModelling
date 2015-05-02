using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using StockLearning.FSharp;
using StockLearning.ModelTraining;

namespace StockModelsConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Starting initialzation process....");

            String dataLocation     = @"C:\Users\Stanley\Documents\Visual Studio 2013\Projects\StockLearning\Stock Data\";
            String tickersFile      = dataLocation + "Tickers.txt";

            
            //read ticker list from the file
            Console.WriteLine("Reading tickers file....");
            var tickers             = DataHelpers.readTickerFile(tickersFile);
            
            //read data for tickers in list from their files
            Console.WriteLine("Reading curves from stock data files....");
            var curves              = DataHelpers.readStockDataFilesFromTickers(dataLocation, tickers);

            //build learning data out of read data
            Console.WriteLine("Building full learning data object....");
            var learningData        = DataHelpers.buildLearningData(curves);

            //get the training data points within the specified date range
            Console.WriteLine("Grabbing training samples from learning data....");
            var trainingPoints      = TrainingData.getTrainingPoints(learningData, DateTime.Parse("07/09/2013"), DateTime.Parse("07/10/2014"));

            //compute the feature and label data for the training points
            Console.WriteLine("Computing feature and label data for training samples....");
            var featureLabelData    = TrainingData.buildModelTrainingData(trainingPoints, learningData);
            double[][] features     = TrainingData.extractModelInputArray(featureLabelData);
            double[][] labels       = TrainingData.extractModelLabelsArray(featureLabelData);

            //build and intialize the neural net trainer
            Console.WriteLine("Creating and intializing neural net....");
            NNTrainer modelTrainer  = new NNTrainer();

            modelTrainer.setFeaturesAndLabels(features, labels);
            modelTrainer.initNet(5, 1);

            int epoch = 0;
            double prevError = 0.0;
            Console.WriteLine("Initialization complete! Starting model training at epoch " + epoch);
            bool continueRunning = true;
            while(!Console.KeyAvailable && continueRunning)
            {
                double error = modelTrainer.runEpoch();
                Console.WriteLine("epoch = " + epoch++ + "avg error = " + error);

                if(epoch > 100 && (Math.Abs(error - prevError) / prevError) < .00001)
                {
                    continueRunning = false;
                }

                prevError = error;
            }

            Console.Read();

        }
    }
}
