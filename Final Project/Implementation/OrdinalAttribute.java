public class OrdinalAttribute extends Attribute {
	ProbabilityCountEnclosure[] possibleValues;
	
	public OrdinalAttribute(String name, int num) {
		super(name, DataType.ORDINAL);
		possibleValues = new ProbabilityCountEnclosure[num];
	}
}
