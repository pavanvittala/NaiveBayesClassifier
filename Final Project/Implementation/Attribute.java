/**
 * Attribute parent class that encodes information about an attribute in a dataset.
 * @author pvittala
 *
 */
public abstract class Attribute {
	public String name;		//The name of the attribute
	public DataType type;	//The type of the attribute {ORDINAL, NOMINAL, CONTINUOUS}
	boolean isClass;		//Boolean that denotes whether or not the attribute is the class/tag of an instance.
	
	public Attribute(String name, DataType type) {
		this.name = name;
		this.type = type;
		this.isClass = false;
	}
}
