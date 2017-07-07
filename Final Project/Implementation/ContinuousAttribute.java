import java.util.HashMap;
/**
 * Class that represents a continous attribute.
 * @author pvittala
 *
 */
public class ContinuousAttribute extends Attribute {
	// There is one ContinuousDistributionEnclosure object per class/tag
	public HashMap<String, ContinuousDistributionEnclosure> distributionInformation;
	//Class/Tag name in String form maps to a CDE object that holds sample mean and variance

	public ContinuousAttribute(String name) {
		super(name, DataType.CONTINUOUS);
		this.distributionInformation = new HashMap<String, ContinuousDistributionEnclosure>();
	}
}
