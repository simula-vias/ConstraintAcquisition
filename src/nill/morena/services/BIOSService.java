package nill.morena.services;

import java.util.ResourceBundle;


public final class BIOSService {

	private  static ResourceBundle BIOS;
	
	static{
		BIOS = ResourceBundle.getBundle("BIOS");
	}
	
	public BIOSService(){
		
//		BIOS = ResourceBundle.getBundle("BIOS");
	}
	
	/**
	 * @return the bIOS
	 */
	public static ResourceBundle getBIOS() {
		return BIOS;
	}

	/**
	 * @param bIOS
	 *            the bIOS to set
	 */
	public void setBIOS(ResourceBundle bIOS) {
		BIOS = bIOS;
	}


}
