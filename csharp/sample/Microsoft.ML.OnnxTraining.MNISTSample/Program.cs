// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) SignalPop LLC. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxTraining;
using MyCaffe.data;

namespace Microsoft.ML.OnnxTraining.MNISTSample
{
    class Program
    {
        static int m_nTrueCount = 0;
        static float m_fTotalLoss = 0;
        static int m_nIterations = 0;
        static List<Tuple<int, float>> m_rgPrecision = new List<Tuple<int, float>>();
        static List<Tuple<int, float>> m_rgAveLoss = new List<Tuple<int, float>>();
        static List<Tuple<byte[], int>> m_rgTrainingData;
        static List<Tuple<byte[], int>> m_rgTestingData;
        static int m_nTrainingDataIdx;
        static int m_nTestingDataIdx;

        static void Main(string[] args)
        {
            string strModel;
            int nBatch;
            int nSteps;

            if (!getArgs(args, out strModel, out nBatch, out nSteps))
                return;

            // Create the training session.
            TrainingSession session = new TrainingSession();

            // Set the error callback function called at the end of the forward pass.
            session.Parameters.OnErrorFunction += OnErrorFunction;
            // Set the evaluation callback function called after the error function on 'DISPLAY_LOSS_STEPS'
            session.Parameters.OnEvaluationFunction += OnEvaluationFunction;
            // Set the training and testing data batch callbacks called to get a new batch of data.
            session.Parameters.OnGetTrainingDataBatch += OnGetTrainingDataBatch;
            session.Parameters.OnGetTestingDataBatch += OnGetTestingDataBatch;

            // Setup the training parameters.
            session.Parameters.SetTrainingParameter(OrtTrainingStringParameter.ORT_TRAINING_MODEL_PATH, strModel);
            session.Parameters.SetTrainingParameter(OrtTrainingStringParameter.ORT_TRAINING_INPUT_LABELS, "labels");
            session.Parameters.SetTrainingParameter(OrtTrainingStringParameter.ORT_TRAINING_OUTPUT_LOSS, "loss");
            session.Parameters.SetTrainingParameter(OrtTrainingStringParameter.ORT_TRAINING_OUTPUT_PREDICTIONS, "predictions");
            session.Parameters.SetTrainingParameter(OrtTrainingStringParameter.ORT_TRAINING_LOG_PATH, "c:\\temp");
            session.Parameters.SetTrainingParameter(OrtTrainingBooleanParameter.ORT_TRAINING_USE_CUDA, true);
            session.Parameters.SetTrainingParameter(OrtTrainingBooleanParameter.ORT_TRAINING_SHUFFLE_DATA, false);
            session.Parameters.SetTrainingParameter(OrtTrainingLongParameter.ORT_TRAINING_EVAL_BATCH_SIZE, nBatch);
            session.Parameters.SetTrainingParameter(OrtTrainingLongParameter.ORT_TRAINING_TRAIN_BATCH_SIZE, nBatch);
            session.Parameters.SetTrainingParameter(OrtTrainingLongParameter.ORT_TRAINING_NUM_TRAIN_STEPS, nSteps);
            session.Parameters.SetTrainingParameter(OrtTrainingLongParameter.ORT_TRAINING_EVAL_PERIOD, 1);
            session.Parameters.SetTrainingParameter(OrtTrainingLongParameter.ORT_TRAINING_DISPLAY_LOSS_STEPS, 400);
            session.Parameters.SetTrainingParameter(OrtTrainingNumericParameter.ORT_TRAINING_LEARNING_RATE, 0.01);
            session.Parameters.SetTrainingOptimizer(OrtTrainingOptimizer.ORT_TRAINING_OPTIMIZER_SGD);
            session.Parameters.SetTrainingLossFunction(OrtTrainingLossFunction.ORT_TRAINING_LOSS_FUNCTION_SOFTMAXCROSSENTROPY);
            session.Parameters.SetupTrainingParameters();

            // Setup the training data information.
            session.Parameters.SetupTrainingData(new List<string>() { "X", "labels" });

            // Load the MNIST dataset from file. See http://yann.lecun.com/exdb/mnist/ to get data files.
            MnistDataLoaderLite dataLoader = new MnistDataLoaderLite("c:\\temp\\data");
            dataLoader.ExtractImages(out m_rgTrainingData, out m_rgTestingData);
            m_nTrainingDataIdx = 0;
            m_nTestingDataIdx = 0;

            // Setup the OnnxRuntime instance.
            OrtEnv env = OrtEnv.Instance();

            // Initialize the training session.
            session.Initialize(env);

            // Run the training session.
            session.RunTraining();
            session.EndTraining();

            // Cleanup.
            session.Dispose();

            Console.WriteLine("Done!");
            Console.WriteLine("press any key to exit...");
            Console.ReadKey();
        }

        private static bool getArgs(string[] args, out string strModel, out int nBatch, out int nSteps)
        {
            strModel = "mnist";
            nBatch = 100;
            nSteps = 2000;

            if (args.Length == 0)
            {
                Console.WriteLine("MNIST Test Application Command Line:");
                Console.WriteLine("--model <model> --batch <batch> --steps <steps>");
                Console.WriteLine(Environment.NewLine);
                Console.WriteLine("Example:");
                Console.WriteLine("--model " + strModel + " --batch " + nBatch.ToString() + " --steps " + nSteps.ToString());
                return false;
            }

            int nIdx = 0;
            while (nIdx < args.Length)
            {
                if (args[nIdx] == "--defaults")
                {
                    Console.WriteLine("Using defaults: --model " + strModel + " --batch " + nBatch.ToString() + " --steps " + nSteps.ToString());
                    return true;
                }

                if (args[nIdx] == "--model")
                {
                    nIdx++;
                    if (nIdx < args.Length)
                        strModel = args[nIdx];

                    nIdx++;
                }
                else if (args[nIdx] == "--batch")
                {
                    nIdx++;
                    if (nIdx < args.Length)
                    {
                        string strBatch = args[nIdx];
                        int.TryParse(strBatch, out nBatch);
                        nIdx++;
                    }
                }
                else if (args[nIdx] == "--steps")
                {
                    nIdx++;
                    if (nIdx < args.Length)
                    {
                        string strSteps = args[nIdx];
                        int.TryParse(strSteps, out nSteps);
                        nIdx++;
                    }
                }
            }

            return true;
        }

        private static void OnGetTestingDataBatch(object sender, DataBatchArgs e)
        {
            loadBatch(e, m_rgTrainingData, ref m_nTrainingDataIdx);
        }

        private static void OnGetTrainingDataBatch(object sender, DataBatchArgs e)
        {
            loadBatch(e, m_rgTestingData, ref m_nTestingDataIdx);
        }

        private static void loadBatch(DataBatchArgs e, List<Tuple<byte[], int>> rgData, ref int nIdx)
        {
            float[] rgRawData = new float[rgData[0].Item1.Length * e.BatchSize];
            float[] rgRawLabels = new float[e.BatchSize * 10];
            List<int> rgDimData = new List<int>() { e.BatchSize };
            List<int> rgDimLabels = new List<int>() { e.BatchSize };
            int nDataOffset = 0;
            int nLabelOffset = 0;

            rgDimData.AddRange(e.InputShape);
            rgDimLabels.AddRange(e.OutputShape);

            for (int i = 0; i < e.BatchSize; i++)
            {
                int nLabel = rgData[nIdx].Item2;
                rgRawLabels[nLabelOffset + nLabel] = 1;

                for (int j = 0; j < rgData[nIdx].Item1.Length; j++)
                {
                    float fVal = rgData[nIdx].Item1[j];

                    // Binarize data
                    if (fVal > 0)
                        fVal = 1.0f;

                    rgRawData[nDataOffset + j] = fVal;
                }

                nDataOffset += rgData[nIdx].Item1.Length;
                nLabelOffset += 10;

                nIdx++;

                if (nIdx == rgData.Count)
                    nIdx = 0;
            }

            DenseTensor<float> tensorData = new DenseTensor<float>(rgRawData, rgDimData.ToArray());
            DenseTensor<float> tensorLabels = new DenseTensor<float>(rgRawLabels, rgDimLabels.ToArray());

            e.Values.Add(NamedOnnxValue.CreateFromTensor<float>("X", tensorData));
            e.Values.Add(NamedOnnxValue.CreateFromTensor<float>("labels", tensorLabels));
        }

        private static void OnErrorFunction(object sender, ErrorFunctionArgs e)
        {
            Tensor<float> labels = e.Find("labels").AsTensor<float>();
            Tensor<float> predict = e.Find("predictions").AsTensor<float>();
            Tensor<float> loss = e.Find("loss").AsTensor<float>();

            if (labels.Dimensions[0] != predict.Dimensions[0])
                throw new Exception("Predict and Labels should have same batch size!");

            if (predict.Dimensions.Length < 2 || predict.Dimensions[1] != 10)
                throw new Exception("The prediction should have at least 2 dimensions.");


            int nBatchSize = predict.Dimensions[0];
            int nOffset = 0;

            for (int n = 0; n < nBatchSize; n++)
            {
                int nPredictMaxIdx = 0;
                float fPredictMax = 0;
                int nLabelMaxIdx = 0;
                float fLabelMax = 0;

                for (int j = 0; j < predict.Dimensions[1]; j++)
                {
                    float fPredict = predict.GetValue(nOffset + j);
                    if (fPredict > fPredictMax)
                    {
                        nPredictMaxIdx = j;
                        fPredictMax = fPredict;
                    }

                    float fLabel = labels.GetValue(nOffset + j);
                    if (fLabel > fLabelMax)
                    {
                        nLabelMaxIdx = j;
                        fLabelMax = fLabel;
                    }
                }

                if (nPredictMaxIdx == nLabelMaxIdx)
                    m_nTrueCount++;

                nOffset += predict.Dimensions[1];
            }

            m_fTotalLoss += loss.GetValue(0);
            m_nIterations++;
        }

        private static void OnEvaluationFunction(object sender, EvaluationFunctionArgs e)
        {
            float fPrecision = (float)m_nTrueCount / e.NumSamples;
            float fAveLoss = m_fTotalLoss / (float)e.NumSamples;

            Console.WriteLine("Step: " + e.Step.ToString() + ", #examples: " + e.NumSamples.ToString() + ", #correct: " + m_nTrueCount.ToString() + ", precision: " + fPrecision.ToString() + ", loss: " + fAveLoss.ToString());

            m_nTrueCount = 0;
            m_fTotalLoss = 0.0f;
        }
    }
}
