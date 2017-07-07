import java.util.HashMap;
/**
 * Class that represents an ordinal attribute.
 * @author pvittala
 *
 */
public class OrdinalAttribute extends Attribute {
	//The possible values an ordinal attribute can take on (in String form) map to a PCE object, which holds information about the counts of each class/tag
	HashMap<String, ProbabilityCountEnclosure> possibleValues;
	
	public OrdinalAttribute(String name, int num) {
		super(name, DataType.ORDINAL);
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
