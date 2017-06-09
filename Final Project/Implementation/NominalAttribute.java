public class NominalAttribute extends Attribute {
	ProbabilityCountEnclosure[] possibleValues;
	
	public NominalAttribute(String name, int num) {
		super(name, DataType.NOMINAL);
		possibleValues = new ProbabilityCountEnclosure[num];
	}
}
