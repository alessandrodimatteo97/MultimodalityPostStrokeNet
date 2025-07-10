import os
import subprocess
import nibabel as nib


def convert_to_niigz(input_file):
    """
    Converts a .nii file to .nii.gz using nibabel, returns the new file path.
    If the file is already .nii.gz or invalid, returns the original path.
    """
    if input_file and input_file.endswith('.nii') and os.path.exists(input_file):
        output_file = input_file + ".gz"
        
        # Avoid reconverting if the compressed version already exists
        if os.path.exists(output_file):
            print(f"✅ File already converted: {output_file}")
            return output_file
        
        print(f"🔄 Converting {input_file} to {output_file}")
        try:
            img = nib.load(input_file)
            nib.save(img, output_file)
            return output_file
        except Exception as e:
            print(f"❌ Error converting {input_file}: {e}")
            return input_file  # fallback to original if conversion fails
    return input_file



def mask_segmentation_with_brainmask(seg_file, brain_mask_file, output_file):
    seg_img = nib.load(seg_file)
    mask_img = nib.load(brain_mask_file)

    seg_data = seg_img.get_fdata()
    mask_data = mask_img.get_fdata()

    masked_seg = seg_data * (mask_data > 0)

    masked_seg_img = nib.Nifti1Image(masked_seg.astype(seg_data.dtype), seg_img.affine, seg_img.header)
    nib.save(masked_seg_img, output_file)

    return output_file



def morph_op(input_filename, output_filename, op):
    cmd1 = f"{ANIMA_FOLDER}/animaMorphologicalOperations -i {input_filename}  -o {output_filename} -a {op}"
    result1 = subprocess.run(cmd1, shell=True)
    if result1.returncode != 0:
        raise RuntimeError(f"Error in animaMorphologicalOperations: {cmd1}")
    return output_filename


def fillHole(input_filename, output_filename):
    cmd1 = f"{ANIMA_FOLDER}/animaFillHoleImage -i {input_filename}  -o {output_filename}"
    result1 = subprocess.run(cmd1, shell=True)
    if result1.returncode != 0:
        raise RuntimeError(f"Error in animaFillHoleImage: {cmd1}")
    return output_filename


def skull_strip(input_filename, output_filename):
    """
    Perform skull stripping using HD-BET.
    """
    os.system(f"hd-bet -i {input_filename} -o {output_filename} -device cuda --save_bet_mask")
    return output_filename


def bias_correct(input_filename, output_filename):
    os.system(f"{ANIMA_FOLDER}/animaN4BiasCorrection -i {input_filename} -o {output_filename}")
    return output_filename


def reorient_RAS(input_filename, output_filename):
    """
    Reorient the input NIfTI file to closest RAS orientation and save as output.
    """
    img = nib.load(input_filename)
    reoriented_img = nib.as_closest_canonical(img)  # Reorient to RAS
    nib.save(reoriented_img, output_filename)
    return output_filename


def register_to_reference(input_filename, output_filename, reference_input, mask_input=None):
    """
    Register an input image to a reference using Anima.
    If a mask is provided, it is also transformed and post-processed (dilation + fill-hole).
    
    Returns:
      (final_image, final_mask) if mask_input is provided,
      otherwise (final_image, None)
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    subject = input_filename.split("/")
    subject_folder = os.path.join(OUTPUT_DIR, subject[2])
    anat_folder = os.path.join(subject_folder, 'anat')
    seg_folder = os.path.join(subject_folder, 'seg')
    
    os.makedirs(anat_folder, exist_ok=True)
    os.makedirs(seg_folder, exist_ok=True)
    
    transform_aff = output_filename.replace('.nii.gz', "_aff.txt")
    transforms_xml = output_filename.replace('.nii.gz', "_transforms.xml")
    
    # We'll build the mask output name only if a mask is provided
    final_mask_filename = None

    print(f"--- Registering {input_filename} to {reference_input} ---")
    
    # 1) Registration
    cmd1 = f"{ANIMA_FOLDER}/animaPyramidalBMRegistration -m {input_filename} -r {reference_input} -o {output_filename} -O {transform_aff} -p 4 -l 2"
    result1 = subprocess.run(cmd1, shell=True)
    if result1.returncode != 0:
        raise RuntimeError(f"Error in animaPyramidalBMRegistration: {cmd1}")
    
    # 2) Convert to XML
    cmd2 = f"{ANIMA_FOLDER}/animaTransformSerieXmlGenerator -i {transform_aff} -o {transforms_xml}"
    result2 = subprocess.run(cmd2, shell=True)
    if result2.returncode != 0:
        raise RuntimeError(f"Error in animaTransformSerieXmlGenerator: {cmd2}")
    
    # 3) If a mask is provided, apply the same transform with nearest neighbor
    if mask_input:
        # By convention, name the initial resampled mask ...
        resampled_mask_output = mask_input.replace('.nii.gz', '_seg.nii.gz')
        # Move from 'anat' to 'seg' folder if needed
        #resampled_mask_output = resampled_mask_output.replace('anat', 'seg')
        
        cmd3 = f"{ANIMA_FOLDER}/animaApplyTransformSerie -i {mask_input} -g {reference_input} -t {transforms_xml} -n nearest -o {resampled_mask_output}"
        result3 = subprocess.run(cmd3, shell=True)
        if result3.returncode != 0:
            raise RuntimeError(f"Error in animaApplyTransformSerie: {cmd3}")
        
        # 4) Morphological operations and fill holes
        #    We define final_mask_filename as the "fill-hole" result
        #dilated_mask = resampled_mask_output.replace('.nii.gz', '_clo.nii.gz')
        #morph_op(resampled_mask_output, dilated_mask, "clos")
        
        final_mask_filename = resampled_mask_output.replace('.nii.gz', '_fh.nii.gz')
        fillHole(resampled_mask_output, final_mask_filename)

    return output_filename, final_mask_filename

def apply_brain_mask(input_image, mask_image, output_image):
    """
    Applica una brain mask a un'immagine anatomica (es. T1, T2, FLAIR).
    """
    print("input image", input_image)
    print("mask image", mask_image)
    img = nib.load(input_image)
    mask = nib.load(mask_image)

    data = img.get_fdata()
    mask_data = mask.get_fdata()

    masked_data = data * (mask_data > 0)

    masked_img = nib.Nifti1Image(masked_data.astype(data.dtype), img.affine, img.header)
    nib.save(masked_img, output_image)
    return output_image



def process_subject(subject_data):
    """
    Process subject data by:
    1. Skull-stripping T1 + bias correction
    2. Register T1 to MNI
    3. Skull-stripping T2/FLAIR + bias correction
    4. Register T2/FLAIR to T1
    5. Reorient final T1/T2/FLAIR and their masks to RAS
    """
    t1_path = convert_to_niigz(subject_data['RawData'].get('T1'))
    t2_path = convert_to_niigz(subject_data['RawData'].get('T2'))
    flair_path = convert_to_niigz(subject_data['RawData'].get('FLAIR'))

    
    t1_mask = subject_data['Derivatives'].get('T1')
    t2_mask = subject_data['Derivatives'].get('T2')
    flair_mask = subject_data['Derivatives'].get('FLAIR')
    
    # We will store final images/masks here and reorient them at the end
    final_outputs = []
    
    if not t1_path:
        print("❌ T1 missing, cannot proceed with registration.")
        return
    
    subject = os.path.basename(os.path.dirname(os.path.dirname(t1_path)))

    anat_folder = os.path.join(OUTPUT_DIR, subject, 'anat')
    seg_folder = os.path.join(OUTPUT_DIR, subject, 'seg')
    
    os.makedirs(anat_folder, exist_ok=True)
    os.makedirs(seg_folder, exist_ok=True)
    
    skull_stripped_t1 = os.path.join(anat_folder, os.path.basename(t1_path).replace('.nii.gz', '_ss.nii.gz'))
    #print(t1_path,"t1_path")
    skull_strip(t1_path, skull_stripped_t1)
    if t1_mask:
        skull_stripped_t1_bet = skull_stripped_t1.replace('.nii.gz', '_bet.nii.gz')
        t1_mask = apply_brain_mask(t1_mask, skull_stripped_t1_bet, os.path.join(seg_folder, os.path.basename(skull_stripped_t1_bet)).replace('.nii.gz', '_hd.nii.gz'))

    n4_output_t1 = skull_stripped_t1.replace('.nii.gz', '_N4.nii.gz')
    bias_correct(skull_stripped_t1, n4_output_t1)

    #--- 2) Register T1 -> MNI
    reg_t1 = n4_output_t1.replace('.nii.gz', '_MNI.nii.gz')
    final_t1, final_t1_mask = register_to_reference(n4_output_t1, reg_t1, MNI_TEMPLATE, t1_mask)
    
    # Keep track for reorientation
    final_outputs.append(final_t1)
    if final_t1_mask:
        final_outputs.append(final_t1_mask)

    #--- 3) T2
    final_t2 = None
    final_t2_mask = None
    if t2_path:
        skull_stripped_t2 = os.path.join(anat_folder, os.path.basename(t2_path).replace('.nii.gz', '_ss.nii.gz'))
        skull_strip(t2_path, skull_stripped_t2)
        
        if t2_mask: 
            skull_stripped_t2_bet = skull_stripped_t2.replace('.nii.gz', '_bet.nii.gz')
            t2_mask = apply_brain_mask(t2_mask, skull_stripped_t2_bet, os.path.join(seg_folder, os.path.basename(skull_stripped_t2_bet)).replace('.nii.gz', '_hd.nii.gz'))


        n4_output_t2 = skull_stripped_t2.replace('.nii.gz', '_N4.nii.gz')
        bias_correct(skull_stripped_t2, n4_output_t2)

        # Register T2 -> T1 (which is already in MNI space)
        reg_t2 = n4_output_t2.replace('.nii.gz', '_T1.nii.gz')
        final_t2, final_t2_mask = register_to_reference(n4_output_t2, reg_t2, final_t1, t2_mask)
        
        final_outputs.append(final_t2)
        if final_t2_mask:
            final_outputs.append(final_t2_mask)

    #--- 4) FLAIR
    final_flair = None
    final_flair_mask = None
    if flair_path:
        skull_stripped_flair = os.path.join(anat_folder, os.path.basename(flair_path).replace('.nii.gz', '_ss.nii.gz'))
        skull_strip(flair_path, skull_stripped_flair)

        if flair_mask: 
            skull_stripped_flair_bet = skull_stripped_flair.replace('.nii.gz', '_bet.nii.gz')
            flair_mask = apply_brain_mask(flair_mask, skull_stripped_flair_bet, os.path.join(seg_folder, os.path.basename(skull_stripped_flair_bet)).replace('.nii.gz', '_hd.nii.gz'))


        n4_output_flair = skull_stripped_flair.replace('.nii.gz', '_N4.nii.gz')
        bias_correct(skull_stripped_flair, n4_output_flair)


        # Register FLAIR -> T1
        reg_flair = n4_output_flair.replace('.nii.gz', '_T1.nii.gz')
        final_flair, final_flair_mask = register_to_reference(n4_output_flair, reg_flair, final_t1, flair_mask)
        
        final_outputs.append(final_flair)
        if final_flair_mask:
            final_outputs.append(final_flair_mask)
    
    #--- 5) Reorient final images + masks to RAS
    for fname in final_outputs:
        if fname and os.path.exists(fname):
            ras_fname = fname.replace('.nii.gz', '_RAS.nii.gz')
            reorient_RAS(fname, ras_fname)
            print(f"Reoriented {fname} to {ras_fname}")

def find_files(base_folder, folder_type):
    """
    Find imaging files in rawdata or derivatives.
    """
    results = {}
    for root, _, files in os.walk(base_folder):
        # For derivatives, you might have a /seg or /anat folder structure.
        # We'll just check if T1, T2, or FLAIR are in the filenames.
        relative_path = os.path.relpath(root, base_folder).replace("/anat", "/seg")
        t1_file = next((f for f in files if 'T1' in f), None)
        t2_file = next((f for f in files if 'T2' in f), None)
        flair_file = next((f for f in files if 'FLAIR' in f), None)
        if t1_file or t2_file or flair_file:
            results[relative_path] = {
                'T1': os.path.join(root, t1_file) if t1_file else None,
                'T2': os.path.join(root, t2_file) if t2_file else None,
                'FLAIR': os.path.join(root, flair_file) if flair_file else None
            }
    return results

def compare_folders(rawdata_folder, derivatives_folder):
    """
    Compare rawdata and derivatives to match images and masks.
    """
    files_rawdata = find_files(rawdata_folder, "rawdata")
    files_derivatives = find_files(derivatives_folder, "derivatives")
    combined_results = {}
    all_subfolders = set(files_rawdata.keys()).union(set(files_derivatives.keys()))
    for subfolder in all_subfolders:
        combined_results[subfolder] = {
            "RawData": files_rawdata.get(subfolder, {"T1": None, "T2": None, "FLAIR": None}),
            "Derivatives": files_derivatives.get(subfolder, {"T1": None, "T2": None, "FLAIR": None}),
        }
    return combined_results

# ------------------------------
# Main execution
# ------------------------------
rawdata_path = "./dataset/rawdata"
derivatives_path = "./dataset/derivatives"




OUTPUT_DIR = "./preprocessed"
MNI_TEMPLATE = "./MNI152_T1_1mm.nii.gz"  # Path to the MNI template
ANIMA_FOLDER = "./anima"

result_dict = compare_folders(rawdata_path, derivatives_path)

for subfolder, data in result_dict.items():
    print(f"Processing {subfolder}...{data}")
    process_subject(data)
