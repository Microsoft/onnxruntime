// Copyright (c) 2018-2020 SignalPop LLC, and contributors. All rights reserved.
// License: Apache 2.0
// License: https://github.com/MyCaffe/MyCaffe/blob/master/LICENSE
// Original Source: https://github.com/MyCaffe/MyCaffe/blob/master/MyCaffe.data/MnistDataLoaderLite.cs
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;

namespace MyCaffe.data
{
    /// <summary>
    /// The MnistDataLoader is used to extrac the MNIST dataset to disk and load the data into the training proces.
    /// </summary>
    /// <remarks>
    /// @see [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
    /// </remarks>
    public class MnistDataLoaderLite
    {
        bool m_bEnableTrace = false;
        string m_strDataPath;
        string m_strTestImagesBin;
        string m_strTestLabelsBin;
        string m_strTrainImagesBin;
        string m_strTrainLabelsBin;
        int m_nChannels = 1;
        int m_nHeight = 0;
        int m_nWidth = 0;

        /// <summary>
        /// The OnProgress event fires during the creation process to show the progress.
        /// </summary>
        public event EventHandler<ProgressArgs> OnProgress;
        /// <summary>
        /// The OnError event fires when an error occurs.
        /// </summary>
        public event EventHandler<ProgressArgs> OnError;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strDataPath">Specifies the location of the data files.</param>
        /// <param name="bEnableTrace">Optionally, specifies whether or not to enable sending output to the Trace (default = false).</param>
        /// <remarks>
        /// Four files make up the MNIST dataset that can be downloaded from http://yann.lecun.com/exdb/mnist/
        ///     t10k-images-idx3-ubyte.gz      ~ testing images
        ///     t10k-labels-idx1-ubyte.gz      ~ testing image labels
        ///     train-images-idx3-ubyte.gz     ~ training images
        ///     train-labels-idx1-ubyte.gz     ~ training image labels
        /// </remarks>
        public MnistDataLoaderLite(string strDataPath, bool bEnableTrace = false)
        {
            m_bEnableTrace = bEnableTrace;
            m_strDataPath = strDataPath;
        }

        /// <summary>
        /// Returns the test images bin filename.
        /// </summary>
        public string TestImagesBinFileName
        {
            get { return m_strTestImagesBin; }
        }

        /// <summary>
        /// Returns the test labels bin filename.
        /// </summary>
        public string TestLabelsBinFileName
        {
            get { return m_strTestLabelsBin; }
        }

        /// <summary>
        /// Returns the train images bin filename.
        /// </summary>
        public string TrainImagesBinFileName
        {
            get { return m_strTrainImagesBin; }
        }

        /// <summary>
        /// Returns the train labels bin filename.
        /// </summary>
        public string TrainLabelsBinFileName
        {
            get { return m_strTrainLabelsBin; }
        }

        /// <summary>
        /// Return the image channel count (should = 1 for black and white images).
        /// </summary>
        public int Channels
        {
            get { return m_nChannels; }
        }

        /// <summary>
        /// Return the image height.
        /// </summary>
        public int Height
        {
            get { return m_nHeight; }
        }

        /// <summary>
        /// Return the image with.
        /// </summary>
        public int Width
        {
            get { return m_nWidth; }
        }

        /// <summary>
        /// Extract the .gz files, expanding them to .bin files.
        /// </summary>
        /// <param name="strDstPath">Specifies the path containing the four MNIST data files.</param>
        public void ExtractFiles(string strDstPath)
        {
            Trace.WriteLine("Unpacking the files");
            m_strTestImagesBin = expandFile(m_strDataPath.TrimEnd('\\') + "\\t10k-images-idx3-ubyte.gz");
            m_strTestLabelsBin = expandFile(m_strDataPath.TrimEnd('\\') + "\\t10k-labels-idx1-ubyte.gz");
            m_strTrainImagesBin = expandFile(m_strDataPath.TrimEnd('\\') + "\\train-images-idx3-ubyte.gz");
            m_strTrainLabelsBin = expandFile(m_strDataPath.TrimEnd('\\') + "\\train-labels-idx1-ubyte.gz");
        }

        private string expandFile(string strFile)
        {
            string strDstFile = strFile + ".bin";
            if (File.Exists(strDstFile))
                return strDstFile;

            FileInfo fi = new FileInfo(strFile);

            using (FileStream fs = fi.OpenRead())
            {
                using (FileStream fsBin = File.Create(strDstFile))
                {
                    using (GZipStream decompStrm = new GZipStream(fs, CompressionMode.Decompress))
                    {
                        decompStrm.CopyTo(fsBin);
                    }
                }
            }

            return strDstFile;
        }

        /// <summary>
        /// Extract the images from the .bin files and save to disk
        /// </summary>
        /// <param name="rgTrainingData">Returns the training data.</param>
        /// <param name="rgTestingData">Returns the testing data.</param>
        public void ExtractImages(out List<Tuple<byte[], int>> rgTrainingData, out List<Tuple<byte[], int>> rgTestingData)
        {
            int nIdx = 0;
            int nTotal = 0;

            try
            {
                ExtractFiles(m_strDataPath);

                reportProgress(nIdx, nTotal, "Creating MNIST images...");

                rgTrainingData = loadFile(m_strTrainImagesBin, m_strTrainLabelsBin, m_strDataPath.TrimEnd('\\') + "\\images_training");
                rgTestingData = loadFile(m_strTestImagesBin, m_strTestLabelsBin, m_strDataPath.TrimEnd('\\') + "\\images_testing");
            }
            catch (Exception excpt)
            {
                reportError(0, 0, excpt);
                throw excpt;
            }
        }

        private List<Tuple<byte[], int>> loadFile(string strImagesFile, string strLabelsFile, string strExportPath)
        {
            if (!Directory.Exists(strExportPath))
                Directory.CreateDirectory(strExportPath);

            Stopwatch sw = new Stopwatch();

            reportProgress(0, 0, "  loading " + strImagesFile + "...");

            BinaryFile image_file = new BinaryFile(strImagesFile);
            BinaryFile label_file = new BinaryFile(strLabelsFile);
            List<Tuple<byte[], int>> rgData = new List<Tuple<byte[], int>>();

            try
            {
                // Verify the files
                uint magicImg = image_file.ReadUInt32();
                uint magicLbl = label_file.ReadUInt32();

                if (magicImg != 2051)
                    throw new Exception("Incorrect image file magic.");

                if (magicLbl != 2049)
                    throw new Exception("Incorrect label file magic.");

                uint num_items = image_file.ReadUInt32();
                uint num_labels = label_file.ReadUInt32();

                if (num_items != num_labels)
                    throw new Exception("The number of items must be equal to the number of labels!");


                // Add the data source to the database.
                uint rows = image_file.ReadUInt32();
                uint cols = image_file.ReadUInt32();

                m_nHeight = (int)rows;
                m_nWidth = (int)cols;

                // Storing to database;
                byte[] rgLabel;
                byte[] rgPixels;

                string strAction = "loading";

                reportProgress(0, (int)num_items, "  " + strAction + " a total of " + num_items.ToString() + " items.");
                reportProgress(0, (int)num_items, "   (with rows: " + rows.ToString() + ", cols: " + cols.ToString() + ")");

                sw.Start();

                for (int i = 0; i < num_items; i++)
                {
                    rgPixels = image_file.ReadBytes((int)(rows * cols));
                    rgLabel = label_file.ReadBytes(1);

                    rgData.Add(new Tuple<byte[], int>(rgPixels, rgLabel[0]));

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        reportProgress(i, (int)num_items, " " + strAction + " data...");
                        sw.Restart();
                    }
                }

                reportProgress((int)num_items, (int)num_items, " " + strAction + " completed.");
            }
            finally
            {
                image_file.Dispose();
                label_file.Dispose();
            }

            return rgData;
        }

        private void reportProgress(int nIdx, int nTotal, string strMsg)
        {
            if (m_bEnableTrace)
            {
                double dfPct = (nTotal == 0) ? 0 : (double)nIdx / (double)nTotal;
                Trace.WriteLine("(" + dfPct.ToString("P") + ") " + strMsg);
            }

            if (OnProgress != null)
                OnProgress(this, new ProgressArgs(new ProgressInfo(nIdx, nTotal, strMsg)));
        }

        private void reportError(int nIdx, int nTotal, Exception err)
        {
            if (m_bEnableTrace)
            {
                double dfPct = (nTotal == 0) ? 0 : (double)nIdx / (double)nTotal;
                Trace.WriteLine("(" + dfPct.ToString("P") + ") ERROR: " + err.Message);
            }

            if (OnError != null)
                OnError(this, new ProgressArgs(new ProgressInfo(nIdx, nTotal, "ERROR", err)));
        }
    }
}
