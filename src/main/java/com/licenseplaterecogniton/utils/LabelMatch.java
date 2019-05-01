/**
 * 
 */
package com.licenseplaterecogniton.utils;

/**
 * @author Arnaud
 *
 */

public class LabelMatch {

	public char matchLetterOrDigits(int predictedClass) {
		char val = ' ';
		
		switch(predictedClass) {
	    
	    case 1:
	    	val = '1';
	    	break;
	    case 2:
	    	val = '2';
	    	break;
	    case 3:
	    	val = '3';
	    	break;
	    case 4:
	    	val = '4';
	    	break;
	    case 5:
	    	val = '5';
	    	break;
	    case 6:
	    	val = '6';
	    	break;
	    case 7:
	    	val = '7';
	    	break;
	    case 8:
	    	val = '8';
	    	break;
	    case 9 :
	    	val = '9';
	    	break;
	    case 10:
	    	val = '0';
	    	break;
	    case 11 :
	    	val = 'A';
	    	break;
	    case 12:
	    	val = 'B';
	    	break;
	    case 13 :
	    	val = 'C';
	    	break;
	    case 14:
	    	val = 'D';
	    	break;
	    case 15:
	    	val = 'E';
	    	break;
	    case 16:
	    	val = 'F';
	    	break;
	    case 17:
	    	val = 'G';
	    	break;
	    case 18:
	    	val = 'H';
	    	break;
	    case 19:
	    	val = 'I';
	    	break;
	    case 20:
	    	val = 'J';
	    	break;
	    case 21:
	    	val = 'K';
	    	break;
	    case 22:
	    	val = 'L';
	    	break;
	    case 23:
	    	val = 'M';
	    	break;
	    case 24:
	    	val = 'N';
	    	break;
	    case 25:
	    	val = 'O';
	    	break;
	    case 26:
	    	val = 'P';
	    	break;
	    case 27:
	    	val = 'Q';
	    	break;
	    case 28:
	    	val = 'R';
	    	break;
	    case 29:
	    	val = 'S';
	    	break;
	    case 30:
	    	val = 'T';
	    	break;
	    case 31:
	    	val = 'U';
	    	break;
	    case 32:
	    	val = 'V';
	    	break;
	    case 33:
	    	val = 'W';
	    	break;
	    case 34:
	    	val = 'X';
	    	break;
	    case 35:
	    	val = 'Y';
	    	break;
	    case 36:
	    	val = 'Z';
	    	break;
	}
	
	return val;
	}
}
