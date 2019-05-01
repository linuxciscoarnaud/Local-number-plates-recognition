/**
 * 
 */
package com.licenseplaterecogniton.impl;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.bytedeco.javacpp.hdf5.*;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;

import org.datavec.image.recordreader.objdetect.ImageObject;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;

/**
 * Label provider for digits and letters recognition, to be used with org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader
 * This label provider reads my dataset that i was able to put together.
 * The dataset contain 36 classes (digits + letters)
 *
 */

/**
 * @author Arnaud
 *
 */
public class NprLabelProvider implements ImageObjectLabelProvider {

	private static DataType refType = new DataType(PredType.STD_REF_OBJ());
    private static DataType charType = new DataType(PredType.NATIVE_CHAR());
    private static DataType intType = new DataType(PredType.NATIVE_INT());
    
    private Map<String, List<ImageObject>> labelMap;
    
    public NprLabelProvider(File dir) throws IOException {
    	labelMap = new HashMap<String, List<ImageObject>>();
    	
    	H5File file = new H5File(dir.getPath() + "/digitStruct.mat", H5F_ACC_RDONLY());
    	Group group = file.openGroup("digitStruct");
    	DataSet nameDataset = group.openDataSet("name");
        DataSpace nameSpace = nameDataset.getSpace();
        DataSet bboxDataset = group.openDataSet("bbox");
        DataSpace bboxSpace = bboxDataset.getSpace();
        long[] dims = new long[2];
        bboxSpace.getSimpleExtentDims(dims);
        int n = (int)(dims[0] * dims[1]);
        
        int ptrSize = Loader.sizeof(Pointer.class);
        PointerPointer namePtr = new PointerPointer(n);
        PointerPointer bboxPtr = new PointerPointer(n);
        nameDataset.read(namePtr, refType);
        bboxDataset.read(bboxPtr, refType);
        
        BytePointer bytePtr = new BytePointer(256);
        PointerPointer topPtr = new PointerPointer(256);
        PointerPointer leftPtr = new PointerPointer(256);
        PointerPointer heightPtr = new PointerPointer(256);
        PointerPointer widthPtr = new PointerPointer(256);
        PointerPointer labelPtr = new PointerPointer(256);
        IntPointer intPtr = new IntPointer(256);
        for (int i = 0; i < n; i++) {
        	DataSet nameRef = new DataSet(file, namePtr.position(i * ptrSize));
            nameRef.read(bytePtr, charType);
            String filename = bytePtr.getString();
            
            Group bboxGroup = new Group(file, bboxPtr.position(i * ptrSize));
            DataSet topDataset = bboxGroup.openDataSet("top");
            DataSet leftDataset = bboxGroup.openDataSet("left");
            DataSet heightDataset = bboxGroup.openDataSet("height");
            DataSet widthDataset = bboxGroup.openDataSet("width");
            DataSet labelDataset = bboxGroup.openDataSet("label");
            
            DataSpace topSpace = topDataset.getSpace();
            topSpace.getSimpleExtentDims(dims);
            int m = (int)(dims[0] * dims[1]);
            ArrayList<ImageObject> list = new ArrayList<ImageObject>(m);
            
            boolean isFloat = topDataset.asAbstractDs().getTypeClass() == H5T_FLOAT;
            if (!isFloat) {
            	topDataset.read(topPtr.position(0), refType);
                leftDataset.read(leftPtr.position(0), refType);
                heightDataset.read(heightPtr.position(0), refType);
                widthDataset.read(widthPtr.position(0), refType);
                labelDataset.read(labelPtr.position(0), refType);
            }
            assert !isFloat || m == 1;
            
            for (int j = 0; j < m; j++) {
            	DataSet topSet = isFloat ? topDataset : new DataSet(file, topPtr.position(j * ptrSize));
                topSet.read(intPtr, intType);
                int top = intPtr.get();
                
                DataSet leftSet = isFloat ? leftDataset : new DataSet(file, leftPtr.position(j * ptrSize));
                leftSet.read(intPtr, intType);
                int left = intPtr.get();

                DataSet heightSet = isFloat ? heightDataset : new DataSet(file, heightPtr.position(j * ptrSize));
                heightSet.read(intPtr, intType);
                int height = intPtr.get();

                DataSet widthSet = isFloat ? widthDataset : new DataSet(file, widthPtr.position(j * ptrSize));
                widthSet.read(intPtr, intType);
                int width = intPtr.get();

                DataSet labelSet = isFloat ? labelDataset : new DataSet(file, labelPtr.position(j * ptrSize));
                labelSet.read(intPtr, intType);
                int label = intPtr.get();
                if (label == 10) {
                    label = 0;
                }
                
                list.add(new ImageObject(left, top, left + width, top + height, Integer.toString(label)));

                topSet.deallocate();
                leftSet.deallocate();
                heightSet.deallocate();
                widthSet.deallocate();
                labelSet.deallocate();
            }
            
            topSpace.deallocate();
            if (!isFloat) {
            	topDataset.deallocate();
                leftDataset.deallocate();
                heightDataset.deallocate();
                widthDataset.deallocate();
                labelDataset.deallocate();
            }
            nameRef.deallocate();
            bboxGroup.deallocate();

            labelMap.put(filename, list);
        }
        
        nameSpace.deallocate();
        bboxSpace.deallocate();
        nameDataset.deallocate();
        bboxDataset.deallocate();
        group.deallocate();
        file.deallocate();
    }
	
	/* (non-Javadoc)
	 * @see org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider#getImageObjectsForPath(java.lang.String)
	 */
	public List<ImageObject> getImageObjectsForPath(String path) {
		// TODO Auto-generated method stub
		File file = new File(path);
        String filename = file.getName();
        return labelMap.get(filename);
	}

	/* (non-Javadoc)
	 * @see org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider#getImageObjectsForPath(java.net.URI)
	 */
	public List<ImageObject> getImageObjectsForPath(URI uri) {
		// TODO Auto-generated method stub
		return getImageObjectsForPath(uri.toString());
	}
}
