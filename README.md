
<body>
  <h1>Audio Classification with SVM</h1>

  <h2>Overview</h2>
  <p>This project implements an audio classification system using Support Vector Machines (SVM) and other machine learning algorithms. The system extracts audio features and evaluates the performance of different classifiers on a given dataset.</p>

  <h2>Dataset</h2>
  <p>The dataset used in this project can be specified using the command line options. Supported datasets include "berlin" and "dafex". Ensure that the dataset is properly formatted and accessible.</p>

  <h2>Requirements</h2>
  <p>To run this project, you need the following libraries:</p>
  <ul>
      <li>pyAudioAnalysis</li>
      <li>scikit-learn</li>
      <li>NumPy</li>
      <li>SciPy</li>
      <li>matplotlib</li>
      <li>seaborn</li>
  </ul>
  <p>You can install the required libraries using pip:</p>
  <pre><code>pip install pyAudioAnalysis scikit-learn numpy scipy matplotlib seaborn</code></pre>

  <h2>Installation</h2>
  <ol>
      <li><strong>Clone the Repository</strong> (if applicable):
          <pre><code>git clone https://github.com/yourusername/audio-classification.git
cd audio-classification</code></pre>
      </li>
      <li><strong>Run the Script</strong>:
          <pre><code>python audio_classification.py --dataset berlin --load_data --extract_features</code></pre>
      </li>
  </ol>

  <h2>Code Explanation</h2>
  <p>The main components of the code are as follows:</p>
  <ul>
      <li><strong>Data Loading:</strong> The dataset is loaded based on the specified dataset type.</li>
      <li><strong>Feature Extraction:</strong> Audio features are extracted using the <code>audioFeatureExtraction</code> module.</li>
      <li><strong>Model Training:</strong> Various classifiers, including SVM, Logistic Regression, Random Forest, KNN, Decision Tree, and Naive Bayes, are trained and evaluated.</li>
      <li><strong>Model Evaluation:</strong> The performance of the models is evaluated using accuracy, precision, recall, and F1-score metrics.</li>
  </ul>

  <h2>Command Line Options</h2>
  <p>The script accepts the following command line options:</p>
  <ul>
      <li><code>-d, --dataset</code>: Specify the dataset type (default: "berlin").</li>
      <li><code>-p, --dataset_path</code>: Specify the path to the dataset.</li>
      <li><code>-l, --load_data</code>: Load the dataset from the specified path.</li>
      <li><code>-e, --extract_features</code>: Extract features from the audio data.</li>
      <li><code>-s, --speaker_indipendence</code>: Enable speaker independence evaluation.</li>
      <li><code>-i, --plot_eigenspectrum</code>: Plot the eigenspectrum of the features.</li>
  </ul>

  <h2>Results</h2>
  <p>The model's performance is evaluated using cross-validation, and the following metrics are reported:</p>
  <ul>
      <li><strong>Accuracy:</strong> The overall accuracy of the model.</li>
      <li><strong>Precision:</strong> The precision for each class.</li>
      <li><strong>Recall:</strong> The recall for each class.</li>
  </ul>

  <h2>Conclusion</h2>
  <p>This project demonstrates the application of machine learning techniques for audio classification. The SVM model, along with other classifiers, provides insights into the effectiveness of different algorithms for this task. Future work could involve hyperparameter tuning and exploring additional features for improved performance.</p>

  <h2>Acknowledgments</h2>
  <ul>
      <li>Thanks to the pyAudioAnalysis library for audio feature extraction.</li>
      <li>Special thanks to the scikit-learn community for their resources and support.</li>
  </ul>

  <h2>License</h2>
  <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>
</body>
