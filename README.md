# AI-supported process monitoring system

Welcome to our Python project! We developed an AI-supported process monitoring system for mechanical engineering as part of a university project. This repository is designed to address advanced data processing tasks using an object-oriented approach for flexibility and scalability. Our project structure is carefully organized to facilitate development, testing, and deployment of machine learning models aimed at signal processing and feature extraction.

## Project Structure

<p>The project is organized into several directories, each serving a specific purpose:</p>

<ul>
  <li><code>modules/</code>: Contains Python modules for data loading, signal preprocessing, feature extraction, model training, and evaluation.</li>
  <li><code>data_loader/</code>: For loading data from various sources.</li>
  <li><code>signal_preprocessor/</code>: Functions for signal preprocessing (filtering, noise reduction).</li>
  <li><code>feature_extractor/</code>: Extracts features from the preprocessed signals.</li>
  <li><code>learner/</code>: Algorithms and methods for training machine learning models.</li>
  <li><code>evaluator/</code>: Functions for evaluating and assessing the trained models.</li>
  <li><code>scripts/</code>: Scripts for specific tasks like data exploration, data extraction, loading experiment data, feature engineering, and model training & evaluation.</li>
  <li><code>test/</code>: Contains test scripts to verify the functionalities of different modules.</li>
  <li><code>.data/</code>: Directory for storing datasets and intermediate results.</li>
  <li><code>env/</code>: Holds the environment configurations required for the project's execution.</li>
  <li><code>artifacts/</code>: Stores results such as plots and model files.</li>
</ul>

<p><code>config.json</code>: Configuration file to control various aspects of the project.</p>
<p><code>LICENSE</code>: Defines the usage rights and restrictions for the project.</p>
<p><code>.gitignore</code>: Specifies intentionally untracked files to ignore.</p>
<p><code>requirements.txt</code>: Lists all Python libraries and packages needed.</p>


# Getting Started

### Prerequisites

<p>Ensure you have Python 3.x installed on your system. You can download it from <a href="https://python.org">python.org</a>.</p>

<p>Before getting started with this project, make sure you have the following Python packages and tools installed:</p>

<ul>
  <li><code>matplotlib</code> (Version 3.8.2)</li>
  <li><code>matplotlib-inline</code> (Version 0.1.6)</li>
  <li><code>numpy</code> (Version 1.26.3)</li>
  <li><code>pandas</code> (Version 2.1.4)</li>
  <li><code>scikit-learn</code> (Version 1.3.2)</li>
  <li><code>scipy</code> (Version 1.11.4)</li>
  <li><code>seaborn</code> (Version 0.13.1)</li>
  <li><code>statsmodels</code> (Version not specified)</li>
  <li><code>pathlib</code> (Version not specified)</li>
  <li><code>tsfresh</code> (Version 0.20.2)</li>
  <li><code>pytest</code> (Version 8.0.0)</li>
</ul>


### Installation

<p>Clone the repository:</p>
<pre><code>git clone https://github.com/link/to/repo</code></pre>
<p>Navigate to the project directory:</p>
<pre><code>cd hsa_python_advanced</code></pre>
<p>Install the required packages:</p>
<pre><code>pip install -r requirements.txt</code></pre>


### Usage

<p>To run the main script and execute the entire pipeline, simply run:</p>
<pre><code>python main.py</code></pre>
<p>For executing specific tasks, navigate to the <code>scripts/</code> directory and run the desired script. For example:</p>
<pre><code>python scripts/data_exploration.py</code></pre>


### Configuration (config.json)

<h2>The <code>config.json</code> file</h2>
<p>The <code>config.json</code> file plays a crucial role in defining the paths to datasets and specifying algorithms used in the project. Here's a breakdown:</p>

<h3>Experiments</h3>
<p>Datasets are organized by experiments, each containing measurements with their respective formats. Adjust the paths as needed when working with your own data.</p>
<ul>
  <li>Experiment 1-3: Data in <code>.tsv</code> and <code>.csv</code> formats.</li>
  <li>Experiment 4: Data serialized in <code>.pkl</code> format.</li>
</ul>
<p>Paths are relative to the <code>.data/</code> directory, ensuring organized and accessible data management.</p>

<h3>Algorithms</h3>
<p>Machine learning algorithms configured for use:</p>
<ul>
  <li>Random Forest: With 1000 estimators and a fixed random state for reproducibility.</li>
  <li>Decision Tree: Configured with a maximum depth to prevent overfitting.</li>
  <li>K-Nearest Neighbors (KNN): Set with 3 neighbors for classification.</li>
</ul>
<p>This structure allows for flexible experimentation with different machine learning strategies and data preprocessing methods.</p>

<p>Please note that this project is configured to be used on univariate time series.</p>

<h3>Testing</h3>
<p>To ensure the reliability of our modules, run the test scripts located in the <code>test/</code> directory:</p>
<pre><code>pytest</code></pre>

<h3>Contributing</h3>
<p>Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.</p>

<h3>License</h3>
<p>This project is licensed under the MIT License - see the LICENSE file for details.</p>

<h3>Acknowledgments</h3>
<p>Thanks to all project members and contributors who have invested their efforts in the development of this project.</p>
