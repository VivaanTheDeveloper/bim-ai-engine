import ifcopenshell
import pandas as pd

def compile_component_schedules(ifc_file_path, output_csv_path):
    """
    Sweeps the BIM model database, extracts metadata traits for every single
    door component, and instantly outputs a structured CSV schedule.
    """
    model = ifcopenshell.open(ifc_file_path)
    doors = model.by_type("IfcDoor")
    
    extracted_schedule_records = []
    
    for idx, door in enumerate(doors):
        # Fallback tracking assignment if specific parameters are left blank by human modelers
        door_tag = door.Name if door.Name else f"D-HOSP-{idx+1:03d}"
        
        width = getattr(door, "OverallWidth", "N/A")
        height = getattr(door, "OverallHeight", "N/A")
        
        # Smart unit check: If raw numbers are tiny decimals (like 0.813m), scale up to mm.
        # If they are already over 10 (like 813), leave them exactly as they are.
        if isinstance(width, (int, float)) and width < 10.0:
            width = width * 1000.0
        if isinstance(height, (int, float)) and height < 10.0:
            height = height * 1000.0
            
        # Access deep property sets for fire safety ratings (Common in Indian Hospital Building Codes)
        fire_rating = "N/A"
        for relDefProperties in door.IsDefinedBy:
            if relDefProperties.is_a("IfcRelDefinesByProperties"):
                property_set = relDefProperties.RelatingPropertyDefinition
                if property_set.is_a("IfcPropertySet") and "FireRating" in property_set.Name:
                    for prop in property_set.HasProperties:
                        if prop.Name == "FireRating":
                            fire_rating = prop.NominalValue.wrappedValue
                            
        extracted_schedule_records.append({
            "Door Mark ID": door_tag,
            "Width (mm)": width,
            "Height (mm)": height,
            "Fire Resistance Rating": fire_rating if fire_rating else "Non-Rated Swing Assembly"
        })
        
    # Compile directly into a data dataframe and export
    df = pd.DataFrame(extracted_schedule_records)
    df.to_csv(output_csv_path, index=False)
    print(f"Schedule extraction pipeline finished. Output file generated at: {output_csv_path}")

if __name__ == "__main__":
    pass