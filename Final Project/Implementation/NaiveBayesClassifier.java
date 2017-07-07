import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Set;

/**
 * 
 * @author Pavan Vittala
 * This is a supervised learning algorithm that implements a Gaussian Naive Bayes classifier. Implemented for CS 484 - at George Mason University
 * This classifier sports the following features:
 * 1. Continuous Attributes' values are estimated using a Normal Distribution. This ensures the independence assumption between attributes.
 * 		a.) Entails calculating sample mean and sample variance for each continuous attribute-class pair
 * 2. Laplace Smoothing ensures that the entire expression doesn't become zero in the case that one of the conditional probabilities is zero.
 * 3. Efficient space complexity: the model stores only occurrence counts and continuous distribution information. The whole dataset does not
 *    need to be loaded into memory.
 * 4. Efficient time complexity: the algorithm's runtime is dependent on the number of attributes, the number of possible values nominal/ ordinal
 *    attributes can take on, and the number of class labels
 *  	a.) Has a worst-case runtime of approximately O(a*p*c), where a = number of attributes and p = the nominal/ ordinal attribute with the
 *          most number of possible values, and c = the number of class labels.
 *      b.) The algorithm can start to get slow if all of these values are high. However, generally speaking, only one or two of these are large,
 *          so our algorithm is relatively efficient.
 *      c.) We could have made our algorithm faster if we'd increased the space complexity. Having a higher run-time complexity was a tradeoff for
 *          having lower space complexity.    	
 *
 */
public class NaiveBayesClassifier {
	private int instanceCount;	//Total number of instances in dataset
	private int instanceCountWithoutMissing;	//Total number of instances if you ignore the instances with missing attribute values
	private int attributeCount;	//Total number of attributes in dataset (doesn't include class attribute)
	private int classCount;		//Total number of possible values for the class
	private int[] classification;	//Count of number of classified instances
	private boolean laplace;
	private String filename;	//Name of the file
	private ArrayList<Attribute> attributeList;
	private long startTime;	//Start time
	private long endTime;	//End time
	private int[][] confusionMatrix;
	
	/**
	 * Creates a NaiveBayesClassifier object. Model is built in the constructor.
	 * @param inputFile: the name of the input file from which to retrieve the data.
	 */
	public NaiveBayesClassifier(String inputFile) {
		this.startTime = System.currentTimeMillis();	//Start time
		//Initialize fields
		this.instanceCount = 0;
		this.laplace = false;
		this.filename = inputFile;
		//Build the model
		buildModel();
	}
	
	/**
	 * Method that is automatically called when the constructor is invoked.
	 * This method builds the model for the NaiveBayesClassifer.
	 * Simply put, loads occurrences of attributes into the appropriate data structures.
	 */
	public void buildModel() {
		System.out.println("Building model for the file: " + this.filename + "\n");
		BufferedReader reader = null;
		try {
			File file = new File(this.filename);
			reader = new BufferedReader(new FileReader(file));
			
			//The first line of the dataset contains the attribute names, separated by ", "
			String attrNames = reader.readLine();	//Read first line
			String[] attr = attrNames.split(", ");	//Split on ", "
			int attrCount = attr.length-1;	//-1 because I need to remove the class from being an attribute
			this.attributeCount = attrCount;
			
			//Second line contains the type (NOMINAL, ORDINAL, CONTINUOUS) of each attribute
			String attrTypes = reader.readLine();
			String[] types = attrTypes.split(", ");
			
			//Third line contains the number of possible values each ordinal or nominal attribute can take
			//For example a nominal attribute "sex" can have the values {MALE, FEMALE}, which is 2 distinct possibilities
			String possibleValCount = reader.readLine();
			String[] valCount = possibleValCount.split(", ");
			int[] valCountInt = new int[valCount.length];
			for (int i = 0; i<valCount.length; i++) {
				int val = Integer.parseInt(valCount[i]);
				valCountInt[i] = val;
			}
			
			//Create an ArrayList and populate it with Attribute objects
			this.attributeList = new ArrayList<Attribute>();
			for (int i = 0; i<attributeCount+1; i++) {
				if (types[i].equals("continuous")) {
					attributeList.add(i, new ContinuousAttribute(attr[i]));
				} else if (types[i].equals("nominal")) {
					attributeList.add(i, new NominalAttribute(attr[i], valCountInt[i]));
				} else if(types[i].equals("ordinal")) {
					attributeList.add(i, new OrdinalAttribute(attr[i], valCountInt[i]));
				}
			}
			
			//Create a reference to the "class" attribute in the attributeList. This is always the last attribute column
			NominalAttribute classAttr = (NominalAttribute) attributeList.get(attributeList.size()-1);
			//Set the number of classes appropriately
			//this.classCount = classAttr.possibleValues.length;
			this.classCount = valCountInt[valCountInt.length-1];
			//That attribute is the class attribute. So set that boolean to true
			attributeList.get(attributeList.size()-1).isClass = true;
			//Initialize confusionMatrix array
			this.confusionMatrix = new int[this.classCount][this.classCount];
			
			//Fourth line in dataset contains the possible values each nominal or ordinal value can take on
			String attrPossibleValues = reader.readLine();
			String[] valueGroupings = attrPossibleValues.split("; ");
			
			//Initialize and populate Nominal and Ordinal attribute possibleValues arrays with ProbabilityCountEnclosures and what not  
			for (int i = 0; i<valueGroupings.length; i++) {
				if (!valueGroupings[i].equals("inf")) {
					String stripped = valueGroupings[i].substring(1, valueGroupings[i].length()-1);
					String[] internalArr = stripped.split(", ");
					if (attributeList.get(i).type.equals(DataType.NOMINAL) || attributeList.get(i).type.equals(DataType.ORDINAL)) {
						if (attributeList.get(i).type.equals(DataType.NOMINAL)) {
							NominalAttribute val = (NominalAttribute) attributeList.get(i);
							for (int j = 0; j<internalArr.length; j++) {
								val.addPossibleValue(internalArr[j]);
							}
						} else if (attributeList.get(i).type.equals(DataType.ORDINAL)) {
							OrdinalAttribute val = (OrdinalAttribute) attributeList.get(i);
							for (int j = 0; j<internalArr.length; j++) {
								val.addPossibleValue(internalArr[j]);
							}
						}
					}
				}
			}
			
			//Start reading the rest of the data from the file
			String line;
			//Number of rows, or instances in the dataset
			int instanceCount = 0;
			while ((line = reader.readLine()) != null) {	//For each row/instance in the dataset
				instanceCount++;
				//What to do in this chunk of code:
				//1.) Populate ProbabilityCountEnclosures in attributeList with values from dataset (for ORDINAL and NOMINAL data values)
				//2.) Calculate the distribution values for continuous attributes
				
				String[] data = line.split(",");
				for (int i = 0; i<data.length; i++) {
					data[i] = data[i].trim();
				}
				
				for (int i = 0; i<data.length; i++) {	//For each column in each row/instance
					if (this.attributeList.get(i).type.equals(DataType.NOMINAL)) {
						NominalAttribute val = (NominalAttribute) this.attributeList.get(i);
						ProbabilityCountEnclosure enc = val.possibleValues.get(data[i]);
						if (enc != null) {
							enc.updateClassCounter(data[data.length-1]);
						}
					} else if (this.attributeList.get(i).type.equals(DataType.ORDINAL)) {
						OrdinalAttribute val = (OrdinalAttribute) this.attributeList.get(i);
						ProbabilityCountEnclosure enc = val.possibleValues.get(data[i]);
						if (enc != null) {
							enc.updateClassCounter(data[data.length-1]);
						}
					} else {	//Continuous Variable
						ContinuousAttribute val = (ContinuousAttribute) this.attributeList.get(i);
						ContinuousDistributionEnclosure check = val.distributionInformation.get(data[data.length-1]);
						if (check == null) {
							val.distributionInformation.put(data[data.length-1], new ContinuousDistributionEnclosure());
						} else {
							check.sampleMean += Double.parseDouble(data[i]);
						}
					}
				}
			}
			//Set instanceCount field
			this.instanceCount = instanceCount;
			
			//Determine if laplace smoothing is required for each ProbabilityCount Enclosure
			for (int i = 0; i<this.attributeList.size()-1 && !this.laplace; i++) {
				if (!this.attributeList.get(i).type.equals(DataType.CONTINUOUS)) {
					if (this.attributeList.get(i).type.equals(DataType.NOMINAL)) {
						NominalAttribute val = (NominalAttribute) this.attributeList.get(i);
						Set<String> keys = val.possibleValues.keySet();
						for (String key : keys) {
							Set<String> keysLev2 = val.possibleValues.get(key).classCounter.keySet();
							for (String key2:keysLev2) {
								int intVal = val.possibleValues.get(key).classCounter.get(key2).intValue();
								if (intVal == 0) {
									this.laplace = true;
								}
							}
						}
					} else if (this.attributeList.get(i).type.equals(DataType.ORDINAL)) {
						OrdinalAttribute val = (OrdinalAttribute) this.attributeList.get(i);
						Set<String> keys = val.possibleValues.keySet();
						for (String key : keys) {
							Set<String> keysLev2 = val.possibleValues.get(key).classCounter.keySet();
							for (String key2:keysLev2) {
								int intVal = val.possibleValues.get(key).classCounter.get(key2).intValue();
								if (intVal == 0) {
									this.laplace = true;
								}
							}
						}
					}
				}
			}
			
			//Calculations needed for normal distribution estimation. Only necessary for continuous variables.
			calculateSampleMean();
			calculateSampleVariance();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				reader.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * Method which calculates the sample mean of the values of the continuous variables in attributeList
	 * sample_mean = (1/n)*SUM[x_i]
	 * The summation has been done in initCountAndDistribution. This method does the division by n.
	 */
	public void calculateSampleMean() {
		NominalAttribute classAttr = (NominalAttribute) attributeList.get(attributeList.size()-1);
		for (int i = 0; i<this.attributeList.size(); i++) {
			if (this.attributeList.get(i).type.equals(DataType.CONTINUOUS)) {
				ContinuousAttribute val = (ContinuousAttribute) attributeList.get(i);
				Set<String> classes = classAttr.possibleValues.keySet();
				for (String key:classes) {
					ContinuousDistributionEnclosure cde = val.distributionInformation.get(key);
					int divisor = classAttr.possibleValues.get(key).classCounter.get(key);
					if (divisor != 0) {
						cde.sampleMean = cde.sampleMean/classAttr.possibleValues.get(key).classCounter.get(key);
					}
				}
			}
		}
	}
	
	/**
	 * Method which calculates the sample variance of the values of the continuous variables in attributeList
	 * Requires calculation of the sample mean first because the formula for sample variance requires the formula for sample mean.
	 * sample_var = (1/n-1)*SUM[(x_i - x_bar)^2]
	 * This requires reading the input file again, unfortunately. Will try to optimize this further at a later point...
	 */
	public void calculateSampleVariance() {
		NominalAttribute classAttr = (NominalAttribute) attributeList.get(attributeList.size()-1);
		BufferedReader reader = null;
		try {
			File file = new File(this.filename);
			reader = new BufferedReader(new FileReader(file));
			skip(reader);
			String line;
			while ((line = reader.readLine()) != null) {
				String[] data = line.split(",");
				for (int i = 0; i<data.length; i++) {
					data[i] = data[i].trim();
				}
				for (int i = 0; i<data.length; i++) {
					if (attributeList.get(i).type.equals(DataType.CONTINUOUS)) {
						ContinuousAttribute val = (ContinuousAttribute) attributeList.get(i);
						double doubleVal = 0.0;
						if (!data[i].equals("?")) {
							doubleVal = Double.parseDouble(data[i]);
						}
						ContinuousDistributionEnclosure cde = val.distributionInformation.get(data[data.length-1]);
						if (data[i].equals("?")) {
							cde.sampleVariance += 0;
						} else {
							cde.sampleVariance += Math.pow(doubleVal-cde.sampleMean, 2);
						}
					}
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				reader.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		for (int i = 0; i<this.attributeList.size(); i++) {
			if (this.attributeList.get(i).type.equals(DataType.CONTINUOUS)) {
				ContinuousAttribute val = (ContinuousAttribute) attributeList.get(i);
				Set<String> keys = val.distributionInformation.keySet();
				for (String key:keys) {
					ContinuousDistributionEnclosure cde = val.distributionInformation.get(key);
					int classAttrCount = classAttr.possibleValues.get(key).classCounter.get(key);
					if (classAttrCount > 1) {
						cde.sampleVariance = cde.sampleVariance/classAttrCount-1;
					}
				}
			}
		}
	}
	
	/**
	 * Used to skip over the first four lines of input files since the first four lines 
	 * house data for the program (attributeNames, attributeTypes, attributeEncodingCount, attributeEncodingPossibilities)
	 * @param reader
	 */
	private void skip(BufferedReader reader) {
		try {
			for (int i = 0; i<4; i++) {
				reader.readLine();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Run a test on the original dataset using the model to see how correctly it classifies each instance.
	 * To clarify, the test set in this method is the training set. Not an optimal way to test a classification algorithm.
	 * If possible use the cross validation test method.
	 * @return
	 */
//	public void testUsingTrainingSet() {
//		System.out.println("Testing model for the file: " + this.filename + "\n");
//		NominalAttribute classAttr = (NominalAttribute) attributeList.get(attributeList.size()-1);
//		BufferedReader reader = null;
//		try {
//			File file = new File(this.filename);
//			reader = new BufferedReader(new FileReader(file));
//			for (int i = 0; i<4; i++) {
//				reader.readLine();
//			}
//			String line;
//			int instances = this.instanceCount;
//			int[] instanceProbabilities = new int[this.classCount];
//			int currentInstance = 0;
//			while ((line = reader.readLine()) != null) {	//"For each instance"
//				currentInstance++;
//				double[] iterationProbs = new double[this.classCount];
//				for (int i = 0; i<iterationProbs.length; i++) {
//					iterationProbs[i] = 1;
//				}
//				String[] data = line.split(",");
//				boolean missing = false;
//				for (int i = 0; i<data.length; i++) {
//					data[i] = data[i].trim();
//					if (data[i].equals("?")) {
//						missing = true;
//					}
//				}
//				if (missing) {	//**Missing value consolidation**
//					instances--;
//					continue;
//				}
//				//System.out.println("Current Instance: " + currentInstance);
//				for (int i = 0; i<data.length-1; i++) {
//					if (this.attributeList.get(i).type.equals(DataType.CONTINUOUS)) {
//						ContinuousAttribute val = (ContinuousAttribute) this.attributeList.get(i);
//						//Use Gaussian Distribution formula:
//						for (int j = 0; j<val.distributionInformation.length; j++) {
//							double probability = gaussianDistributionFormula(Double.parseDouble(data[i]), val.distributionInformation[j].sampleMean, val.distributionInformation[j].sampleVariance);
//							//System.out.println("This continuous attribute ("+ data[i]+") is contributing: " + probability);
//							iterationProbs[j]*= probability; 
//						}
//					} else if (this.attributeList.get(i).type.equals(DataType.NOMINAL)) {
//						NominalAttribute val = (NominalAttribute) attributeList.get(i);
//						if (this.laplace) {
//							for (int j = 0; j<val.possibleValues.length; j++) {
//								if (data[i].equals(val.possibleValues[j].ordinalOrNominalName)) {
//									for (int k = 0; k<val.possibleValues[j].classCounter.length; k++) {
//										double numerator = val.possibleValues[j].classCounter[k]+1;
//										double denominator = classAttr.possibleValues[k].classCounter[k]+this.classCount;
//										double probability = numerator/denominator;
//										//System.out.println("This nominal attribute ("+ data[i]+") is contributing: " + probability);
//										iterationProbs[k]*= probability;
//									}
//									break;
//								}
//							}
//						} else {
//							for (int j = 0; j<val.possibleValues.length; j++) {
//								if (data[i].equals(val.possibleValues[j].ordinalOrNominalName)) {
//									for (int k = 0; k<val.possibleValues[j].classCounter.length; k++) {
//										double numerator = val.possibleValues[j].classCounter[k];
//										double denominator = classAttr.possibleValues[k].classCounter[k];
//										double probability = numerator/denominator; 
//										//System.out.println("This nominal attribute ("+ data[i]+") is contributing: " + probability);
//										iterationProbs[k]*= probability;
//									}
//									break;
//								}
//							}
//						}
//					} else {	//Ordinal
//						OrdinalAttribute val = (OrdinalAttribute) this.attributeList.get(i);
//						if (this.laplace) {
//							for (int j = 0; j<val.possibleValues.length; j++) {
//								if (data[i].equals(val.possibleValues[j].ordinalOrNominalName)) {
//									for (int k = 0; k<val.possibleValues[j].classCounter.length; k++) {
//										double numerator = val.possibleValues[j].classCounter[k]+1;
//										double denominator = classAttr.possibleValues[k].classCounter[k]+this.classCount;
//										double probability = numerator/denominator; 
//										//System.out.println("This ordinal attribute ("+ data[i]+") is contributing: " + probability);
//										iterationProbs[k]*= probability;
//									}
//									break;
//								}
//							}
//						} else {
//							for (int j = 0; j<val.possibleValues.length; j++) {
//								if (data[i].equals(val.possibleValues[j].ordinalOrNominalName)) {
//									for (int k = 0; k<val.possibleValues[j].classCounter.length; k++) {
//										double numerator = val.possibleValues[j].classCounter[k];
//										double denominator = classAttr.possibleValues[k].classCounter[k];
//										double probability = numerator/denominator; 
//										//System.out.println("This ordinal attribute ("+ data[i]+") is contributing: " + probability);
//										iterationProbs[k]*= probability;
//									}
//									break;
//								}
//							}
//						}
//					}
//				}
//				//System.out.println("Probabilities: ");
//				/*
//				for (int i = 0; i<iterationProbs.length; i++) {
//					System.out.println(classAttr.possibleValues[i].ordinalOrNominalName + ": " + iterationProbs[i]);
//				}
//				*/
//				double max = iterationProbs[0];
//				int retIndex = 0;
//				for (int i = 0; i<iterationProbs.length; i++) {
//					if (iterationProbs[i] > max) {
//						max = iterationProbs[i];
//						retIndex = i;
//					}
//				}
//				
//				int row = 0;
//				int column = 0;
//				column = retIndex;
//				for (int i = 0; i<classAttr.possibleValues.length; i++) {
//					if (classAttr.possibleValues[i].ordinalOrNominalName.equals(data[data.length-1])) {
//						row = i;
//					}
//				}
//				confusionMatrix[row][column]++;
//				
//				//System.out.print("Current Instance was classified as: ");
//				if (classAttr.possibleValues[retIndex].ordinalOrNominalName.equals(data[data.length-1])) {
//					instanceProbabilities[retIndex]++;
//					//System.out.println(classAttr.possibleValues[retIndex].ordinalOrNominalName);
//				}
//			}
//			this.classification = instanceProbabilities;
//			this.instanceCountWithoutMissing = instances;
//		} catch (IOException e) {
//			e.printStackTrace();
//		} finally {
//			try {
//				reader.close();
//			} catch (IOException e) {
//				e.printStackTrace();
//			}
//		}
//		present();
//	}
	
	/**
	 * Test using n-fold cross validation. Much better method than testing using the training set.
	 * @param folds
	 */
	public void testCrossValidation(int folds) {
		//TODO
	}
	
	public double gaussianDistributionFormula(double value, double sampleMean, double sampleVariance) {
		double inverseSqrt = 1;
		if (sampleVariance != 0) {
			inverseSqrt = (1/Math.sqrt(2*Math.PI*sampleVariance));
		}
		double ePowerNumerator = Math.pow(value-sampleMean, 2);
		double ePowerDenominator = 1;
		if (sampleVariance != 0) {
			ePowerDenominator = 2*sampleVariance;
		}
		double ePower = Math.pow(Math.E, -ePowerNumerator/ePowerDenominator);
		return inverseSqrt*ePower;
	}
	
//	public String present() {
//		StringBuilder sb = new StringBuilder();
//		int sum = 0;
//		sb.append("Test Results:").append(System.lineSeparator()).append("-------------").append(System.lineSeparator()).append(System.lineSeparator());
//		sb.append("Class Breakdown:").append(System.lineSeparator()).append("----------------").append(System.lineSeparator());
//		for (int i = 0; i<this.classification.length; i++) {
//			sb.append(((NominalAttribute) this.attributeList.get(this.attributeList.size()-1)).possibleValues[i].ordinalOrNominalName).append(": ").append(this.classification[i]).append(System.lineSeparator());
//			sum+=this.classification[i];
//		}
//		sb.append(System.lineSeparator());
//		NominalAttribute classAttr = (NominalAttribute) this.attributeList.get(this.attributeList.size()-1);
//		sb.append("Confusion Matrix:").append(System.lineSeparator()).append("-----------------").append(System.lineSeparator());
//		for (int i = 0; i<this.confusionMatrix.length; i++) {
//			if (i != this.confusionMatrix.length-1) {
//				sb.append(String.format("%-6s", classAttr.possibleValues[i].ordinalOrNominalName));
//			} else {
//				sb.append(String.format("%-3s", classAttr.possibleValues[i].ordinalOrNominalName));
//			}
//		}
//		sb.append(" <-- Classified As");
//		sb.append(System.lineSeparator());
//		for (int i = 0; i<this.confusionMatrix.length; i++) {
//			sb.append(String.format("%-6s", "-"));
//		}
//		sb.append(System.lineSeparator());
//		for (int i = 0; i<this.confusionMatrix.length; i++) {
//			for (int j = 0; j<this.confusionMatrix[i].length; j++) {
//				if (j != this.confusionMatrix[j].length-1) {
//					sb.append(String.format("%-6s", this.confusionMatrix[i][j]));
//				} else {
//					sb.append(String.format("%-6s", this.confusionMatrix[i][j])).append("| ");
//				}
//			}
//			if (i == 0) {
//				sb.append(classAttr.possibleValues[i].ordinalOrNominalName).append(" <-- Actual Class").append(System.lineSeparator());
//			} else {
//				sb.append(classAttr.possibleValues[i].ordinalOrNominalName).append(System.lineSeparator());
//			}
//		}
//		sb.append(System.lineSeparator());
//		
//		sb.append("Overall Results:").append(System.lineSeparator()).append("----------------").append(System.lineSeparator());
//		sb.append("Total Correctly Classified: ").append(sum).append(System.lineSeparator());
//		sb.append("Total Instances In Dataset Without Missing Values: ").append(this.instanceCountWithoutMissing).append(System.lineSeparator());
//		sb.append("Versus ... Total Instances In Dataset: ").append(this.instanceCount).append(System.lineSeparator());
//		double fractionCorrect = ((double)sum/this.instanceCountWithoutMissing);
//		double fractionIncorrect = 1-fractionCorrect;
//		sb.append("Percentage correctly classified (total correctly classified/total in dataset without missing values): ").append(fractionCorrect*100).append("%").append(System.lineSeparator());
//		sb.append("Percentage incorrectly classified (total incorrectly classified/total in dataset without missing values): ").append(fractionIncorrect*100).append("%").append(System.lineSeparator());;
//		this.endTime = System.currentTimeMillis();
//		sb.append("Elapsed time for test: ").append((double) (this.endTime-this.startTime)/1000).append(" seconds").append(System.lineSeparator());
//		sb.append("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------").append(System.lineSeparator());
//		return sb.toString();
//	}

	
	/**
	 * This method prints the data in the attributeList field. HashMaps and all.
	 * @return Constructed String 
	 */
	public String printModel() {
		StringBuilder sb = new StringBuilder();
		//Test to see if model was built correctly using print statements and stuff.
		for (int i = 0; i<this.attributeList.size(); i++) {
			if (this.attributeList.get(i).type.equals(DataType.CONTINUOUS)) {
				ContinuousAttribute val = (ContinuousAttribute) this.attributeList.get(i);
				sb.append(val.name.toUpperCase()).append(System.lineSeparator());
				Set<String> keys = val.distributionInformation.keySet();
				for (String key:keys) {
					ContinuousDistributionEnclosure enc = val.distributionInformation.get(key);
					sb.append(key).append(System.lineSeparator());
					sb.append("\tSample mean: ").append(enc.sampleMean).append(System.lineSeparator());
					sb.append("\tSample variance: ").append(enc.sampleVariance).append(System.lineSeparator());
				}
				sb.append(System.lineSeparator()).append(System.lineSeparator());
			} else if (this.attributeList.get(i).type.equals(DataType.NOMINAL)) {
				NominalAttribute val = (NominalAttribute) this.attributeList.get(i);
				sb.append(val.name.toUpperCase()).append(System.lineSeparator());
				Set<String> keys = val.possibleValues.keySet();
				for (String key:keys) {
					ProbabilityCountEnclosure pce = val.possibleValues.get(key);
					sb.append(pce.ordinalOrNominalName).append(":").append(System.lineSeparator());
					Set<String> classKeys = pce.classCounter.keySet();
					for (String classKey:classKeys) {
						int mappedInt = pce.classCounter.get(classKey);
						sb.append("\t").append(classKey).append(" count of ").append(mappedInt).append(System.lineSeparator());
					}
				}
				sb.append(System.lineSeparator()).append(System.lineSeparator());
			} else {	//Ordinal
				OrdinalAttribute val = (OrdinalAttribute) this.attributeList.get(i);
				sb.append(val.name.toUpperCase()).append(System.lineSeparator());
				Set<String> keys = val.possibleValues.keySet();
				for (String key:keys) {
					ProbabilityCountEnclosure pce = val.possibleValues.get(key);
					sb.append(pce.ordinalOrNominalName).append(":").append(System.lineSeparator());
					Set<String> classKeys = pce.classCounter.keySet();
					for (String classKey:classKeys) {
						int mappedInt = pce.classCounter.get(classKey);
						sb.append("\t").append(classKey).append(" count of ").append(mappedInt).append(System.lineSeparator());
					}
				}
				sb.append(System.lineSeparator()).append(System.lineSeparator());
			}
		}
		return sb.toString();
	}
	
//	public static void run() {
//		//Test 1: adult
//				NaiveBayesClassifier adult = new NaiveBayesClassifier("adult.txt");
//				System.out.println(adult.printModel());
//				adult.testUsingTrainingSet();
//				System.out.println(adult.present());
//				
//				//Test 2: Car evaluation
//				NaiveBayesClassifier car = new NaiveBayesClassifier("car.txt");
//				System.out.println(car.printModel());
//				car.testUsingTrainingSet();
//				System.out.println(car.present());
//				
//				//Test 3: Connect-4
//				NaiveBayesClassifier connect = new NaiveBayesClassifier("connect-4.txt");
//				System.out.println(connect.printModel());
//				connect.testUsingTrainingSet();
//				System.out.println(connect.present());
//				
//				//Test 4: Breast-cancer-wisconsin
//				NaiveBayesClassifier breast_cancer = new NaiveBayesClassifier("breast-cancer-wisconsin-preproc.txt");
//				System.out.println(breast_cancer.printModel());
//				breast_cancer.testUsingTrainingSet();
//				System.out.println(breast_cancer.present());
//				
//				//Test 5: Contraceptive Method Choice
//				NaiveBayesClassifier cmc = new NaiveBayesClassifier("cmc.txt");
//				System.out.println(cmc.printModel());
//				cmc.testUsingTrainingSet();
//				System.out.println(cmc.present());
//				
//				//Test 6: Poker Hand
//				NaiveBayesClassifier poker = new NaiveBayesClassifier("poker-hand-training-true.txt");
//				System.out.println(poker.printModel());
//				poker.testUsingTrainingSet();
//				System.out.println(poker.present());
//				
//				//Test 7: Abalone
//				NaiveBayesClassifier abalone = new NaiveBayesClassifier("abalone.txt");
//				System.out.println(abalone.printModel());
//				abalone.testUsingTrainingSet();
//				System.out.println(abalone.present());
//				
//				//Test 8: Covtype
//				NaiveBayesClassifier covtype = new NaiveBayesClassifier("covtype.txt");
//				System.out.println(covtype.printModel());
//				covtype.testUsingTrainingSet();
//				System.out.println(covtype.present());
//
//				//Test 9: Bank-marketing
//				NaiveBayesClassifier bank = new NaiveBayesClassifier("bank-full.txt");
//				System.out.println(bank.printModel());
//				bank.testUsingTrainingSet();
//				System.out.println(bank.present());
//				
//				//Test 10: Ads
//				NaiveBayesClassifier ad = new NaiveBayesClassifier("ad.txt");
//				System.out.println(ad.printModel());
//				ad.testUsingTrainingSet();
//				System.out.println(ad.present());
//	}
	
	
	/**
	 * BEGIN - Getters
	 */
	public int getInstanceCount() {
		return instanceCount;
	}

	public int getInstanceCountWithoutMissing() {
		return instanceCountWithoutMissing;
	}

	public int getAttributeCount() {
		return attributeCount;
	}

	public int getClassCount() {
		return classCount;
	}

	public int[] getClassification() {
		return classification;
	}

	public boolean isLaplace() {
		return laplace;
	}

	public String getFilename() {
		return filename;
	}

	public ArrayList<Attribute> getAttributeList() {
		return attributeList;
	}

	public long getStartTime() {
		return startTime;
	}

	public long getEndTime() {
		return endTime;
	}

	public int[][] getConfusionMatrix() {
		return confusionMatrix;
	}
	
	/**
	 * END - Getters
	 */

	/**
	 * Test the NaiveBayesClassifier using various datasets
	 * @param args
	 */
	public static void main(String[] args) {
		//run();
		NaiveBayesClassifier adult = new NaiveBayesClassifier("bin/adult.txt");
		System.out.println(adult.printModel());
		//File currentDir = new File("").getAbsoluteFile();
		//System.out.println(currentDir.getPath());
	}
}
