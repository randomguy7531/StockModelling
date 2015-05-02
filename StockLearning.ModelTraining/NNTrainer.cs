using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AForge.Neuro;
using AForge.Neuro.Learning;

namespace StockLearning.ModelTraining
{
    public class NNTrainer
    {
        private double[][] m_featureVectors;
        private double[][] m_labelVectors;
        private ActivationNetwork m_network;
        private BackPropagationLearning m_teacher;

        public NNTrainer()
        {

        }

        public void setFeaturesAndLabels(double[][] features, double[][] labels)
        {
            m_featureVectors = features;
            m_labelVectors = labels;
        }
        public void initNet(int firstLayerNeurons, int secondLayerNeurons)
        {
            if(m_featureVectors.Length == 0)
            {
                throw new InvalidOperationException("Cannot init net while feature vector is not initialized!");
            }

            int inputLayerNeurons = m_featureVectors[0].Length;

            m_network = new ActivationNetwork(new SigmoidFunction(), inputLayerNeurons, firstLayerNeurons, secondLayerNeurons);

            m_teacher = new BackPropagationLearning(m_network);
        }

        public double runEpoch()
        {
            double errorSum = 0;
            for(int i = 0; i < m_featureVectors.Length; i++)
            {
                errorSum += m_teacher.Run(m_featureVectors[i], m_labelVectors[i]);
            }
            return errorSum / m_featureVectors.Length;
        }
    }
}
