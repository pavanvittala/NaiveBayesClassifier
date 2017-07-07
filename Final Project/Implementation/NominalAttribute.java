import java.util.HashMap;
/**
 * Class that represents a nominal attribute.
 * Note: the class/tag attribute is represented as a nominal attribute. This is definitely not the best way to do this... but for now it works.
 * @author pvittala
 *
 */
public class NominalAttribute extends Attribute {
	//The possible values a nominal attribute can take on (in String form) map to a PCE object, which holds information about the counts of each class/tag
	HashMap<String, ProbabilityCountEnclosure> possibleValues;
	
	public NominalAttribute(String name, int num) {
		super(name, DataType.NOMINAL);
		possibleValues = new HashMap<String, ProbabilityCountEnclosure>();
	}
	
	/**
	 * Add another possible value that a nominal attribute can take on to the HashMap. 
	 * @param value
	 * @return
	 */
	public boolean addPossibleValue(String value) {
		ProbabilityCountEnclosure check = possibleValues.get(value);
		if (check == null) {
			possibleValues.put(value, new ProbabilityCountEnclosure(value));
			return true;
		} else {
			return false;
		}
		
	}
}
