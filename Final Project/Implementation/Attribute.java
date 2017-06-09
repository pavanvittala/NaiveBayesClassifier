
public abstract class Attribute {
	public String name;
	public DataType type;
	boolean isClass;
	
	public Attribute(String name, DataType type) {
		this.name = name;
		this.type = type;
		this.isClass = false;
	}
}
