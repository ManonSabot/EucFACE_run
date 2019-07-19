;*******************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
;*******************************************

begin

case_name = (/\
              "/short/w35/mm3972/cable/runs/Marks_latest_branch_with_fixes/GSWP3_gw_off_unify_para/Outputs-standard/gw_off",\
              "/short/w35/mm3972/cable/runs/Marks_latest_branch_with_fixes/GSWP3_gw_off_unify_para_redistrb-on/Outputs-standard/gw_off",\
              "/short/w35/mm3972/cable/runs/Marks_latest_branch_with_fixes/GSWP3_gw_off_unify_para/Outputs-Haverd2013/gw_off",\
              "/short/w35/mm3972/cable/runs/Marks_latest_branch_with_fixes/GSWP3_gw_on_unify_para/Outputs-standard/gw_on",\
              "/short/w35/mm3972/cable/runs/Marks_latest_branch_with_fixes/GSWP3_gw_on_no_aquifer_influence_unify_para/Outputs-standard/gw_on",\
              "/short/w35/mm3972/cable/runs/Marks_latest_branch_with_fixes/GSWP3_gw_on_unify_para/Outputs-Haverd2013/gw_on",\
              "/short/w35/mm3972/cable/runs/Marks_latest_branch_with_fixes/GSWP3_gw_on_no_aquifer_influence_unify_para/Outputs-Haverd2013/gw_on"\
             /)

case_name_brf = (/\
              "off-std",\
              "off-redstrb-std",\
              "off-Hvrd",\
              "on-std",\
              "on_norchrg-std",\
              "on-Hvrd",\
              "on_norchrg-Hvrd"\
               /)


var_name= (/\
           "SoilMoist",\     ;"m^3/m^3"  Average layer soil moisture
           "GWMoist",\       ;"mm3/mm3" Aquifer mositure content, Groundwater_off doesn't have
           "CanopInt",\      ;"kg/m^2", "Canopy intercepted water storage"
           "SWE",\           ;"kg/m^2", "Snow water equivalent"
           "GWdz"\           ;"Mean aquifer layer thickness" ;
         	 /)

Soil_thickness = (/0.022, 0.058, 0.154, 0.409, 1.085, 2.872/) ; 6 soil layers and 1 aquifer

; ____________________________ Options _______________________________
deseason = True
print(deseason)
scale = "World" ;"AU" or "World"
PRE   = False ; True using Precipitation-Runoff-Evaporation vs GRACE
             ; False using SoilMoist+GWMoist+CanopInt+SWE vs GRACE
GRACE_opt = "CLASS" ; "Sanaa"

;______________________________ Data _________________________________
year_s = 2003
year_e = 2009
year_sum = year_e-year_s+1
var_sum  = dimsizes(var_name)

GRACE      = new((/84,360,720/),float)
TWS_200212 = new((/7,360,720/),float)
TWS        = new((/7,84,360,720/),float)

GRACE      = 0.0
TWS_200212 = 0.0
TWS        = 0.0

; ___________________________ Plot Check ______________________________
pic = "check"
wks = gsn_open_wks("pdf",pic)
gsn_define_colormap(wks,"ViBlGrWhYeOrRe")
res = True
;res@cnFillMode         = "RasterFill"            ; Raster Mode
;res@cnFillOn           = True                            ;��ɫ
res@tmBorderThicknessF = 3.0
;res@cnLinesOn          = False                       ; �رյ�ֵ������
res@gsnDraw            = True;False  ; Don't draw plots
res@gsnFrame           = True;False  ; ����ҳ

;_______________________________ PFT ___________________________________
filename = case_name(0)+"/cable_out_"+year_s+".nc"
f = addfile (filename,"r")
iveg = f->iveg
delete(f)
delete(filename)

;______________________________ GRACE __________________________________
counter = 0

if GRACE_opt .eq. "CLASS" then
   GRACE_data = "/short/w35/mm3972/cable/runs/Marks_latest_branch_with_fixes/plots/CLASS_v1/"
   do year = year_s,year_e
     if ( mod(year,4) .eq. 0) then
        dom = (/31,29,31,30,31,30,31,31,30,31,30,31/)
     else
        dom = (/31,28,31,30,31,30,31,31,30,31,30,31/)
     end if
   
     filename = GRACE_data+"CLASS_v1.1_"+year+".nc"  
     f = addfile (filename,"r")
     do month = 0,11
        GRACE(counter,:,:) = lonFlip(f->dw(month,:,:))*3600*24*dom(month)
        counter = counter +1
     end do ;month
   end do ;year

else if GRACE_opt .eq. "Sanaa" then

   GRACE_data = "/short/w35/mm3972/cable/runs/Marks_latest_branch_with_fixes/plots/GRACE/"
   do year = year_s,year_e
      filename = GRACE_data+"GRACE_ds_"+year+".nc"
      f = addfile (filename,"r")
      do month = 0,11
         GRACE(counter,:,:) = lonFlip(f->dw(month,:,:))*10.
         counter = counter +1
      end do ;month
   end do ;year

end if
end if


GRACE@_FillValue = f->dw@_FillValue

delete(counter)
delete(f)


;______________________________ INPUT __________________________________
do case_num = 0,6
print(case_name(case_num))
counter = 0

; read in 2002-12
filename = case_name(case_num)+"/cable_out_2002.nc"
f = addfile (filename,"r")
if PRE .eq. False then
  if case_num .le. 2 then
    TWS_200212(case_num,:,:) = f->SWE(11,:,:) + f->CanopInt(11,:,:)
  else
    TWS_200212(case_num,:,:) = f->GWMoist(11,:,:)*f->GWdz*1000. + f->SWE(11,:,:) + f->CanopInt(11,:,:)
  end if
  do layer = 0,5
    TWS_200212(case_num,:,:) = TWS_200212(case_num,:,:) + f->SoilMoist(11,layer,:,:)*Soil_thickness(layer)*1000.
  end do
end if

; _______________________ for check _______________________
;filename = case_name(case_num)+"/cable_out_2003.nc"
;f = addfile (filename,"r")
;;TWS_200212(case_num,:,:) = f->SWE(0,:,:) + f->CanopInt(0,:,:) - TWS_200212(case_num,:,:)
;TWS_200212(case_num,:,:) = f->SWE(1,:,:) + f->CanopInt(1,:,:) - (f->SWE(0,:,:) + f->CanopInt(0,:,:))
;do layer = 0,5
;;  TWS_200212(case_num,:,:) = TWS_200212(case_num,:,:) + f->SoilMoist(0,layer,:,:)*Soil_thickness(layer)*1000.
;  TWS_200212(case_num,:,:) = TWS_200212(case_num,:,:) + (f->SoilMoist(1,layer,:,:)-f->SoilMoist(0,layer,:,:))*Soil_thickness(layer)*1000.
;end do
;
;res@cnLevelSelectionMode = "ExplicitLevels"
;levels = (/-500.,-400,-300.,-200,-100.,-50,50,100,200,300,400,500/)
;res@cnLevels = levels

;plot = gsn_csm_contour_map(wks,TWS_200212(case_num,:,:),res)
;
;TWS_200212(case_num,:,:) = TWS_200212(case_num,:,:) - (f->Rainf(0,:,:) - f->Evap(0,:,:) - f->Qs(0,:,:) - f->Qsb(0,:,:))*3600.*24.*31.
;plot = gsn_csm_contour_map(wks,TWS_200212(case_num,:,:),res)
;
;TWS_200212(case_num,:,:) = (f->Rainf(0,:,:) - f->Evap(0,:,:) - f->Qs(0,:,:) - f->Qsb(0,:,:))*3600.*24.*31.
;plot = gsn_csm_contour_map(wks,TWS_200212(case_num,:,:),res)
;___________________________________________________________

; read in 2003-2009
do year = year_s,year_e
  print("year = " +year)
  if ( mod(year,4) .eq. 0) then
     dom = (/31,29,31,30,31,30,31,31,30,31,30,31/)
  else
     dom = (/31,28,31,30,31,30,31,31,30,31,30,31/)
  end if

  filename = case_name(case_num)+"/cable_out_"+year+".nc"
  f = addfile (filename,"r")

  do month = 0,11
    if PRE .eq. True then
      TWS(case_num,counter,:,:) = (f->Rainf(month,:,:) - f->Evap(month,:,:) - f->Qs(month,:,:) - f->Qsb(month,:,:)) *3600*24*dom(month)
    else
      if case_num .le. 2 then
        TWS(case_num,counter,:,:) = f->SWE(month,:,:) + f->CanopInt(month,:,:)
      else
        TWS(case_num,counter,:,:) =  f->SWE(month,:,:) + f->CanopInt(month,:,:) ;f->GWMoist(month,:,:)*f->GWdz*1000. +
      end if
      do layer = 0,5
        TWS(case_num,counter,:,:) = TWS(case_num,counter,:,:) + f->SoilMoist(month,layer,:,:)*Soil_thickness(layer)*1000.
      end do
    end if
    counter = counter + 1
  end do ;month
end do ;year

end do ;case

GRACE!0 = "time"
GRACE!1 = "lat"
GRACE!2 = "lon"
GRACE&lat = f->Qle&y
GRACE&lon = f->Qle&x

; Calculate TWS_change
TWS_change = new((/7,84,360,720/),float)
TWS_change = 0.0

if PRE .eq. True then
  TWS_change = TWS
else
  TWS_change(:,0,:,:) = TWS(:,0,:,:) - TWS_200212(:,:,:)
  TWS_change(:,1:83,:,:) = TWS(:,1:83,:,:) - TWS(:,0:82,:,:)
end if

;delete(TWS)
delete(TWS_200212)

print("point 1 ")

TWS_change!0 = "case"
TWS_change!1 = "time"
TWS_change!2 = "lat"
TWS_change!3 = "lon"
TWS_change&lat = f->Qle&y
TWS_change&lon = f->Qle&x
TWS_change@_FillValue   = f->Qle@_FillValue

tmp   = new((/7,84/),float)
tmp   = 0.0
tmp!0 = "case"
tmp!1 = "time"
tmp@_FillValue   = f->Qle@_FillValue
do time = 0,83
do case_num = 0,6
   tmp(case_num,time) = avg(TWS(case_num,time,:,:))
end do
end do
x = ispan(1,84,1)
plot = gsn_csm_xy (wks,x,tmp,res)
;plot = gsn_csm_contour_map(wks,dim_avg_n_Wrap(GRACE(:,:,:),0),res)
;plot = gsn_csm_contour(wks,dim_avg_n_Wrap(GRACE(:,:,:),0)*1.0,res)
print("point 2 ")

;do case_num = 0,6
;plot = gsn_csm_contour_map(wks,dim_avg_n_Wrap(TWS_change(case_num,:,:,:),0),res)
;plot = gsn_csm_contour(wks,dim_avg_n_Wrap(TWS_change(case_num,:,:,:),0)*1.0,res)
;end do
print("point 3 ")


;**************** ͼ�������� *****************
pres                    = True                                      ; ���� panel resources.
pres@gsnMaximize        = True
pres@gsnPanelLabelBar   = False                    ; ���� panel labelbar.
pres@cnLevelSelectionMode = "ExplicitLevels" ;"ManualLevels"
pres@lbBoxLinesOn       = True                       ;�ر�lbar box ����
pres@lbTitleFont        = 25
pres@lbLabelFont        = 25
pres@lbLabelBarOn       = False ; True                    ; �رո���plot��labelbar
pres@lbTitleFontHeightF = 0.013
pres@lbLabelFontHeightF = 0.013
pres@lbAutoManage       = False

;plot_s = new(2,graphic)

;res@cnLevelSelectionMode = "ExplicitLevels"
;levels = (/-100.,-80,-60.,-40,-20.,-10,10,20,40,60,80,100/)
;res@cnLevels = levels

;do month = 0,11
;pres@txString = "month " + (month+1)
;plot_s(0) = gsn_csm_contour(wks,GRACE(month,:,:)*1.0,res)
;plot_s(1) = gsn_csm_contour(wks,TWS_change(0,month,:,:)*1.0,res)
;gsn_panel(wks,(/plot_s/),(/1,2/),pres)
;end do
;delete(levels)
;delete(pres)

; __________________________ Deseasonalize _____________________________
; rmMonAnnCycTLL : Removes the annual cycle from "monthly" data.
if deseason .eq. True then
  print(deseason)

  GRACE = rmMonAnnCycTLL(GRACE) ; dimensioned time x lat x lon

;  plot = gsn_csm_contour_map(wks,dim_avg_n_Wrap(GRACE(:,:,:),0),res)
;  plot = gsn_csm_contour(wks,dim_avg_n_Wrap(GRACE(:,:,:),0)*1.0,res)

  do case_num = 0,6
    TWS_change(case_num,:,:,:) = rmMonAnnCycTLL(TWS_change(case_num,:,:,:))
;    plot = gsn_csm_contour_map(wks,dim_avg_n_Wrap(TWS_change(case_num,:,:,:),0),res)
;    plot = gsn_csm_contour(wks,dim_avg_n_Wrap(TWS_change(case_num,:,:,:),0)*1.0,res)
  end do
end if

delete(res)
delete(pic)
delete(wks)

; _____________________  CALCULATE CORRELATION ________________________
GRACE_TWS  = new((/7,360,720/),float)
GRACE_TWS  = 0.0

do case_num = 0,6
GRACE_TWS(case_num,:,:) = escorc(TWS_change(case|case_num,lat|:,lon|:,time|:), GRACE(lat|:,lon|:,time|:))
; escorc: Computes the (Pearson) sample linear cross-correlations at lag 0 only.
; The rightmost dimension is usually time.
end do
GRACE_TWS!0 = "case"
GRACE_TWS!1 = "lat"
GRACE_TWS!2 = "lon"
GRACE_TWS&lat = f->Qle&y
GRACE_TWS&lon = f->Qle&x
delete(TWS_change)
delete(GRACE)

;_____________________________ PLOT ___________________________________
if deseason .eq. True then
   pic = GRACE_opt+"-GRACE_TWS_correl_"+scale+"_deseasonal_"+year_s+"-"+year_e
else
   pic = GRACE_opt+"-GRACE_TWS_correl_"+scale+"_"+year_s+"-"+year_e
end if

wks = gsn_open_wks("pdf",pic)
gsn_define_colormap(wks,"WhiteBlueGreenYellowRed")

res                    = True
res@cnFillMode         = "RasterFill"            ; Raster Mode
res@cnFillOn           = True                            ;��ɫ
res@tmBorderThicknessF = 3.0

res@gsnDraw            = False  ; Don't draw plots
res@gsnFrame           = False  ; ����ҳ
res@lbLabelBarOn       = True

;************** ����labelbar ***************
res@lbBoxLinesOn       = True                       ;�ر�lbar box ����
res@lbTitleFont        = 25
res@lbLabelFont        = 25
res@lbTitleFontHeightF = 0.013
res@lbLabelFontHeightF = 0.013
res@txString           = ""
res@tmXBLabelFont      = 25 ;Sets the font index for the bottom X-Axis labels.
res@tmYLLabelFont      = 25

;*************** ���õ�ֵ�� *****************
res@cnLinesOn          = False                       ; �رյ�ֵ������
res@cnLineColor        = "black"
res@cnLineThicknessF   = 1.5
res@cnLineLabelsOn     = False
res@gsnMaximize        = True
res@cnExplicitLabelBarLabelsOn = True   ;?
res@cnLevelSelectionMode = "ExplicitLevels"

;************ ����������ֵ��Χ **************
res@tmXBLabelFontThicknessF = 0.015
res@tmYLLabelFontThicknessF = 0.015
res@tmXBLabelFontHeightF = 0.015
res@tmYLLabelFontHeightF = 0.015

if scale .eq. "World" then
   res@trYMinF            = -60
   res@trYMaxF            = 80
   res@mpMinLatF          = -60
   res@mpMaxLatF          = 80
   res@mpCenterLonF       = 160

   res@tmYLMode  = "Explicit"             ;��Y����������������,��ʾ�Լ���Ҫ��ֵ
   res@tmYLValues=(/60,30,0,-30,-60/)
   res@tmYLLabels=(/"60~S~o~N~N","30~S~o~N~N","0~S~o~N~","30~S~o~N~S","60~S~o~N~S"/)
   res@tmXBMode  = "Explicit"
   res@tmXBValues=(/240,300,0,60,120,180/)
   res@tmXBLabels=(/"120~S~o~N~W","60~S~o~N~W","0~S~o~N~","60~S~o~N~E","120~S~o~N~E","180~S~o~N~"/)
   
else if scale .eq. "AU" then

   res@gsnAddCyclic       = False
   res@trYMinF            = -40
   res@trYMaxF            = -27
   res@mpMinLatF          = -40
   res@mpMaxLatF          = -27
   res@trXMinF            = 140
   res@trXMaxF            = 155
   res@mpMinLonF          = 140
   res@mpMaxLonF          = 155

   res@tmYLMode  = "Explicit"             ;��Y����������������,��ʾ�Լ���Ҫ��ֵ
   res@tmYLValues=(/-30,-40/)
   res@tmYLLabels=(/"30~S~o~N~S","40~S~o~N~S"/)
   res@tmXBMode  = "Explicit"
   res@tmXBValues=(/140,150/)
   res@tmXBLabels=(/"140~S~o~N~E","150~S~o~N~E"/)
end if
end if

;**************** ͼ�������� *****************
pres                    = True                                      ; ���� panel resources.
pres@gsnMaximize        = True
pres@gsnPanelLabelBar   = False                    ; ���� panel labelbar.
pres@cnLevelSelectionMode = "ExplicitLevels" ;"ManualLevels"
pres@lbBoxLinesOn       = True                       ;�ر�lbar box ����
pres@lbTitleFont        = 25
pres@lbLabelFont        = 25
pres@lbLabelBarOn       = False ; True                    ; �رո���plot��labelbar
pres@lbTitleFontHeightF = 0.013
pres@lbLabelFontHeightF = 0.013
pres@lbAutoManage       = False
pres@lbTitleDirection   = "Across"             ; title direction
pres@lbTitlePosition    = "Bottom";Right"
;pres@pmLabelBarSide     = "Bottom";"Right"
;pres@pmLabelBarOrthogonalPosF = -0.021
;pres@pmLabelBarParallelPosF = 0.

plots = new(7,graphic)

; ____________________ all PFT __________________________
print("for all PFTs")
do case_num = 0,6

res@tiMainString = case_name_brf(case_num)

if scale .eq. "World" then
   print(case_name_brf(case_num)+" "+avg(GRACE_TWS(case_num,:,:)))
   levels = (/0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0/)
   res@cnLevels = levels
   plots(case_num) = gsn_csm_contour_map(wks,GRACE_TWS(case_num,:,:),res)
else if scale .eq. "AU" then
   print(case_name_brf(case_num)+" "+avg(GRACE_TWS(case_num,{-40:-27},{140:155})))
   levels = (/0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6/)
   res@cnLevels = levels
   plots(case_num) = gsn_csm_contour_map(wks,GRACE_TWS(case_num,{-40:-27},{140:155}),res)
end if
end if

end do
gsn_panel(wks,(/plots/),(/4,2/),pres)

; tmp var
tmp   = new((/7,360,720/),float)
tmp   = 0.0
tmp!0 = "case"
tmp!1 = "lat"
tmp!2 = "lon"
tmp&lat = f->Qle&y
tmp&lon = f->Qle&x
tmp@_FillValue   = f->Qle@_FillValue

; ____________________ forest __________________________
print("for forest")
pres@txString = "forest"
do case_num = 0,6
  tmp(case_num,:,:) = where(iveg .ge. 1 .and. iveg .le. 4, GRACE_TWS(case_num,:,:), GRACE_TWS@_FillValue )
  res@tiMainString = case_name_brf(case_num)
  if scale .eq. "World" then
    print(case_name_brf(case_num)+" "+avg(tmp(case_num,:,:)))
    levels = (/0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0/)
    res@cnLevels = levels
    plots(case_num) = gsn_csm_contour_map(wks,tmp(case_num,:,:),res)
  else if scale .eq. "AU" then
    print(case_name_brf(case_num)+" "+avg(tmp(case_num,{-40:-27},{140:155})))
    levels = (/0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6/)
    res@cnLevels = levels
    plots(case_num) = gsn_csm_contour_map(wks,tmp(case_num,{-40:-27},{140:155}),res)
  end if
  end if
end do
gsn_panel(wks,(/plots/),(/4,2/),pres)


; ____________________ grass+shrub+crop __________________________
print("for grass+shrub+crop")
pres@txString = "grass+shrub+crop"
do case_num = 0,6
  tmp(case_num,:,:) = where(iveg .ge. 5 .and. iveg .le. 11, GRACE_TWS(case_num,:,:), GRACE_TWS@_FillValue )
  res@tiMainString = case_name_brf(case_num)
  if scale .eq. "World" then
    print(case_name_brf(case_num)+" "+avg(tmp(case_num,:,:)))
    levels = (/0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0/)
    res@cnLevels = levels
    plots(case_num) = gsn_csm_contour_map(wks,tmp(case_num,:,:),res)
  else if scale .eq. "AU" then
    print(case_name_brf(case_num)+" "+avg(tmp(case_num,{-40:-27},{140:155})))
    levels = (/0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6/)
    res@cnLevels = levels
    plots(case_num) = gsn_csm_contour_map(wks,tmp(case_num,{-40:-27},{140:155}),res)
  end if
  end if
end do
gsn_panel(wks,(/plots/),(/4,2/),pres)

end
