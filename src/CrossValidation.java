import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Utils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.core.converters.ConverterUtils.DataSource;
/**
 * Performs a single run of cross-validation.
 *
 * Command-line parameters:
 * <ul>
 *    <li>-t filename - the dataset to use</li>
 *    <li>-x int - the number of folds to use</li>
 *    <li>-s int - the seed for the random number generator</li>
 *    <li>-c int - the class index, "first" and "last" are accepted as well;
 *    "last" is used by default</li>
 *    <li>-W classifier - classname and options, enclosed by double quotes; 
 *    the classifier to cross-validate</li>
 * </ul>
 *
 * Example command-line:
 * <pre>
 * java CrossValidation -t anneal.arff -c last -x 10 -s 1 -W "weka.classifiers.trees.J48 -C 0.25"
 * </pre>
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public class CrossValidation {

  /**
   * Performs the cross-validation. See Javadoc of class for information
   * on command-line parameters.
   *
   * @param args        the command-line parameters
   * @throws Excecption if something goes wrong
   */
  public static void main(String[] args) throws Exception {
    // loads data and set class index
    
	  BufferedReader reader = new BufferedReader(
              new FileReader("arff/australian.arff"));
Instances data = new Instances(reader);
reader.close();
	 
	  // setting class attribute if the data format does not provide this information
	  // For example, the XRFF format saves the class attribute information as well
	  if (data.classIndex() == -1)
	    data.setClassIndex(data.numAttributes() - 1);
	  J48 j48 = new J48();
	  j48.setUnpruned(true);  

	  Evaluation eval = new Evaluation(data);
	  //eval.crossValidateModel(j48, data, 10, new Random(1));

	  
	  Classifier cls = new J48();
	  cls.buildClassifier(data);
	  // evaluate classifier and print some statistics
	  eval.evaluateModel(cls, data);
	  System.out.println(eval.toSummaryString("\nResults\n======\n", false));
	  
	  
	  
	  
    // perform cross-validation    // output evaluation
    System.out.println();
    System.out.println("=== Setup ===");
    System.out.println("Dataset: " + data.relationName());
    System.out.println();
  }
}