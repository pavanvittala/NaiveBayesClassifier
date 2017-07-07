import java.util.HashMap;
/**
 * Nominal/Ordinal Attribute -> PossibleValue -> PCE (HashMap<Class/Tag, Counter>) -> Class/Tag -> Counter
 * @author pvittala
 *
 */
public class ProbabilityCountEnclosure {
	String ordinalOrNominalName;
	HashMap<String, Integer> classCounter;
	
	public ProbabilityCountEnclosure(String name) {
		this.ordinalOrNominalName = name;
		//totalCount = 0;
		this.classCounter = new HashMap<String, Integer>();
	}
	
	public void updateClassCounter(String className) {
		Integer check = this.classCounter.get(className);
		if (check == null) {
			this.classCounter.put(className, 1);
		} else {
			int oldCount = check.intValue();
			this.classCounter.put(className, oldCount+1);
		}
	}
}
