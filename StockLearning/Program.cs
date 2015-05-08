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
            String modelsLocation   = @"C:\Users\Stanley\Documents\Visual Studio 2013\Projects\StockLearning\Models\";
            String tickersFile      = dataLocation + "Tickers2.txt";
                        
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
            var trainingPoints      = TrainingValidationData.getTrainingValidationPoints(learningData, DateTime.Parse("07/09/2013"), DateTime.Parse("07/10/2014"));

            //compute the feature and label data for the training points
            Console.WriteLine("Computing feature and label data for training samples....");
            var featureLabelData    = TrainingValidationData.buildModelTrainingData(trainingPoints, learningData);
            Console.Write("features.....");
            var features            = TrainingValidationData.extractModelInputArray(featureLabelData);
            Console.Write("done\r\nlabels.....");
            var labels              = TrainingValidationData.extractModelLabelsArray(featureLabelData);
            Console.Write("done\r\n");

            //build and intialize the neural net trainer
            Console.WriteLine("Creating and intializing neural net....");
            NNTrainer modelTrainer  = new NNTrainer();
            modelTrainer.setTrainingFeaturesAndLabels(features, labels);
            modelTrainer.initNet(3, 0, 1);

            int epoch = 0;
            int epochLimit = 1000000;
            double prevError = 0.0;

            Console.WriteLine("Initialization complete! Starting model training at epoch " + epoch);
            bool continueRunning = true;

            while (continueRunning)
            {
                while (!Console.KeyAvailable && continueRunning)
                {
                    double error = modelTrainer.runEpoch();
                    Console.WriteLine("epoch = " + epoch++ + "avg error = " + error);

                    if (epoch > epochLimit && (Math.Abs(error - prevError) / prevError) < .004001)
                    {
                        continueRunning = false;
                    }

                    prevError = error;
                }

                String input = "q";

                Console.WriteLine("Epoch limit reached, enter command within 10 second timeout: \r\n\t" + 
                                    "q or enter:  + stop training and proceed on to validation \r\n\t" + 
                                    "<some integer> -> enter: run more training iterations");

                try
                {
                    input = Reader.ReadLine(10000);
                }
                catch(TimeoutException)
                {
                    Console.WriteLine("Timeout limit reached....proceeding with validation computation....");
                }

                if (input != "q")
                {
                    continueRunning = true;
                    int newParsedLimit = 0;

                    bool parseSuccess = int.TryParse(input, out newParsedLimit);

                    if(parseSuccess)
                    {
                        epochLimit += newParsedLimit;
                        continueRunning = true;
                    }
                    else
                    {
                        epochLimit += epochLimit;
                        continueRunning = true;
                    }
                }
                else
                {
                    continueRunning = false;
                }
            }

            //write the model to a file
            Console.WriteLine("Writing model to file....");
            Guid modelGuid = Guid.NewGuid();
            String dateTimeString = DateTime.Now.ToString("yyyyMMdd_hhmmss");
            String modelFileName = dateTimeString + "_" + modelGuid.ToString() + ".nn";
            modelTrainer.writeModelToFile(modelsLocation + modelFileName);

            //get the raw output of the training samples post training for later analysis and write to file.
            Console.WriteLine("Writing training results to file....");
            List<double[][]> trainingResults = modelTrainer.GetTrainingOutputs();
            String trainingResultsFileName = dateTimeString + "_" + modelGuid.ToString() + "_trainResults.csv";
            modelTrainer.WriteResultsToFile(trainingResults, modelsLocation + trainingResultsFileName);

            //get the validation data points within the specified date range
            Console.WriteLine("Grabbing validation samples from learning data....");
            var validationPoints = TrainingValidationData.getTrainingValidationPoints(learningData, DateTime.Parse("07/11/2014"), DateTime.Parse("01/11/2015"));
      
            //compute the feature and label data for the training points
            Console.WriteLine("Computing feature and label data for validation samples....");
            var featureLabelDataValidation = TrainingValidationData.buildModelTrainingData(validationPoints, learningData);
            Console.Write("features.....");
            var validationFeatures = TrainingValidationData.extractModelInputArray(featureLabelDataValidation);
            Console.Write("done\r\nlabels.....");
            var validationLabels = TrainingValidationData.extractModelLabelsArray(featureLabelDataValidation);
            Console.Write("done\r\n");

            //get the results from the validation data
            Console.WriteLine("Computing results of validation data....");
            List<double[][]> validationResults = modelTrainer.GetValidationOutputs(validationFeatures, validationLabels);

            //output the validation results to a file
            Console.WriteLine("Writing validation results to file....");
            String validationResultsFileName = dateTimeString + "_" + modelGuid.ToString() + "_validateResults.csv";
            modelTrainer.WriteResultsToFile(validationResults, modelsLocation + validationResultsFileName);

            Console.WriteLine("Done....");
            //pause before closing
            Console.Read();
        }
    }
}
