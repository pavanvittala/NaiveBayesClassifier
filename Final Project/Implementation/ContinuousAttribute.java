public class ContinuousAttribute extends Attribute {
	public ContinuousDistributionEnclosure[] distributionInformation; // There is one ContinuousDistributionEnclosure object per class

	public ContinuousAttribute(String name) {
		super(name, DataType.CONTINUOUS);
	}
}
