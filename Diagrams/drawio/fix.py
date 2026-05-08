import io

file_path = r'd:\SCS\المستوي الرابع - الترم الأول\مشروع التخرج\AIBusterArt\Diagrams\drawio\component_diagram( simpler).drawio'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_content = lines[:73]

new_xml = """        <!-- ============ Data/Control Flow Edges ============ -->
        <mxCell id="e5" value="feeds" style="endArrow=block;endFill=1;strokeColor=#60a8d0;strokeWidth=1.5;fontSize=11;" edge="1" source="c-preproc" target="c-tta" parent="1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <mxCell id="e6" value="grayscale" style="endArrow=block;endFill=1;strokeColor=#60a8d0;strokeWidth=1.5;fontSize=11;" edge="1" source="c-preproc" target="c-fft" parent="1">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="325" y="250" />
            </Array>
          </mxGeometry>
        </mxCell>

        <mxCell id="e7" value="RGB+JPEG" style="endArrow=block;endFill=1;strokeColor=#60a8d0;strokeWidth=1.5;fontSize=11;" edge="1" source="c-preproc" target="c-ela" parent="1">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="325" y="250" />
            </Array>
          </mxGeometry>
        </mxCell>

        <mxCell id="e8" value="RGB" style="endArrow=block;endFill=1;strokeColor=#60a8d0;strokeWidth=1.5;fontSize=11;" edge="1" source="c-preproc" target="c-noise" parent="1">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="325" y="250" />
            </Array>
          </mxGeometry>
        </mxCell>

        <mxCell id="e9" value="3 views" style="endArrow=block;endFill=1;strokeColor=#4a2080;strokeWidth=1.5;fontSize=11;" edge="1" source="c-tta" target="c-vit" parent="1">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="650" y="350" />
            </Array>
          </mxGeometry>
        </mxCell>

        <mxCell id="e10" value="3 views" style="endArrow=block;endFill=1;strokeColor=#9060c0;strokeWidth=1.5;fontSize=11;" edge="1" source="c-tta" target="c-siglip" parent="1">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="650" y="350" />
            </Array>
          </mxGeometry>
        </mxCell>

        <mxCell id="e11" value="3 views" style="endArrow=block;endFill=1;strokeColor=#9060c0;strokeWidth=1.5;fontSize=11;" edge="1" source="c-tta" target="c-smogy" parent="1">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="650" y="350" />
            </Array>
          </mxGeometry>
        </mxCell>

        <!-- ============ IWebUI ============ -->
        <mxCell id="sock-webui" value="" style="shape=requiredInterface;html=1;direction=east;strokeColor=#1F3864;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="200" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="ball-webui" value="IWebUI" style="shape=ellipse;html=1;fillColor=#dce8fb;strokeColor=#4a72c4;strokeWidth=2;labelPosition=center;verticalLabelPosition=bottom;align=center;verticalAlign=top;spacingTop=5;fontStyle=1;fontSize=11;" vertex="1" parent="1">
          <mxGeometry x="205" y="125" width="10" height="10" as="geometry" />
        </mxCell>
        <mxCell id="link-browser-webui" value="uses" style="endArrow=none;strokeColor=#1F3864;strokeWidth=1.5;dashed=1;fontSize=11;" edge="1" source="c-browser" target="sock-webui" parent="1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="link-gradio-webui" style="endArrow=none;strokeColor=#4a72c4;strokeWidth=1.5;" edge="1" source="c-gradio" target="ball-webui" parent="1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <!-- ============ IAnalyze ============ -->
        <mxCell id="sock-analyze" value="" style="shape=requiredInterface;html=1;direction=east;strokeColor=#1a5c35;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="460" y="120" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="ball-analyze" value="IAnalyze" style="shape=ellipse;html=1;fillColor=#fff3e8;strokeColor=#f09040;strokeWidth=2;labelPosition=center;verticalLabelPosition=bottom;align=center;verticalAlign=top;spacingTop=5;fontStyle=1;fontSize=11;" vertex="1" parent="1">
          <mxGeometry x="465" y="125" width="10" height="10" as="geometry" />
        </mxCell>
        <mxCell id="link-android-analyze" value="POST /analyze" style="endArrow=none;strokeColor=#1a5c35;strokeWidth=1.5;dashed=1;fontSize=11;" edge="1" source="c-android" target="sock-analyze" parent="1">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="430" y="410" />
              <mxPoint x="430" y="130" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="link-gradio-analyze" value="mounted on" style="endArrow=none;strokeColor=#4a72c4;strokeWidth=1.5;fontSize=11;" edge="1" source="c-gradio" target="sock-analyze" parent="1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="link-fastapi-analyze" style="endArrow=none;strokeColor=#f09040;strokeWidth=1.5;" edge="1" source="c-fastapi" target="ball-analyze" parent="1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <!-- ============ IPreprocess ============ -->
        <mxCell id="sock-preproc" value="" style="shape=requiredInterface;html=1;direction=south;strokeColor=#f09040;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="490" y="180" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="ball-preproc" value="IPreprocess" style="shape=ellipse;html=1;fillColor=#ddeeff;strokeColor=#60a8d0;strokeWidth=2;labelPosition=center;verticalLabelPosition=bottom;align=center;verticalAlign=top;spacingTop=5;fontStyle=1;fontSize=11;" vertex="1" parent="1">
          <mxGeometry x="495" y="190" width="10" height="10" as="geometry" />
        </mxCell>
        <mxCell id="link-fastapi-preproc" value="delegates" style="endArrow=none;strokeColor=#f09040;strokeWidth=1.5;fontSize=11;" edge="1" source="c-fastapi" target="sock-preproc" parent="1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="link-preproc-ball" style="endArrow=none;strokeColor=#60a8d0;strokeWidth=1.5;" edge="1" source="c-preproc" target="ball-preproc" parent="1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <!-- ============ IScore ============ -->
        <mxCell id="sock-score" value="" style="shape=requiredInterface;html=1;direction=north;strokeColor=#4a2080;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="490" y="715" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="ball-score" value="IScore" style="shape=ellipse;html=1;fillColor=#d8c0f0;strokeColor=#4a2080;strokeWidth=2;labelPosition=center;verticalLabelPosition=top;align=center;verticalAlign=bottom;spacingBottom=5;fontStyle=1;fontSize=11;" vertex="1" parent="1">
          <mxGeometry x="495" y="720" width="10" height="10" as="geometry" />
        </mxCell>
        <mxCell id="link-ens-score" style="endArrow=none;strokeColor=#4a2080;strokeWidth=1.5;" edge="1" source="c-ens" target="sock-score" parent="1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="link-noise-score" value="score" style="endArrow=none;strokeColor=#40a070;strokeWidth=1.5;fontSize=11;" edge="1" source="c-noise" target="ball-score" parent="1">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="325" y="725" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="link-vit-score" value="score" style="endArrow=none;strokeColor=#4a2080;strokeWidth=1.5;fontSize=11;" edge="1" source="c-vit" target="ball-score" parent="1">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="650" y="725" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="link-siglip-score" value="score" style="endArrow=none;strokeColor=#9060c0;strokeWidth=1.5;fontSize=11;" edge="1" source="c-siglip" target="ball-score" parent="1">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="650" y="725" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="link-smogy-score" value="score" style="endArrow=none;strokeColor=#9060c0;strokeWidth=1.5;fontSize=11;" edge="1" source="c-smogy" target="ball-score" parent="1">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="650" y="725" />
            </Array>
          </mxGeometry>
        </mxCell>

        <!-- ============ IVerdict ============ -->
        <mxCell id="sock-verdict" value="" style="shape=requiredInterface;html=1;direction=west;strokeColor=#f09040;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="660" y="770" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="ball-verdict" value="IVerdict" style="shape=ellipse;html=1;fillColor=#d8c0f0;strokeColor=#4a2080;strokeWidth=2;labelPosition=center;verticalLabelPosition=bottom;align=center;verticalAlign=top;spacingTop=5;fontStyle=1;fontSize=11;" vertex="1" parent="1">
          <mxGeometry x="655" y="775" width="10" height="10" as="geometry" />
        </mxCell>
        <mxCell id="link-fastapi-verdict" style="endArrow=none;strokeColor=#f09040;strokeWidth=1.5;dashed=1;" edge="1" source="c-fastapi" target="sock-verdict" parent="1">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="800" y="130" />
              <mxPoint x="800" y="780" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="link-ens-verdict" style="endArrow=none;strokeColor=#4a2080;strokeWidth=1.5;" edge="1" source="c-ens" target="ball-verdict" parent="1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <!-- ============ IModelWeights ============ -->
        <mxCell id="sock-weights" value="" style="shape=requiredInterface;html=1;direction=east;strokeColor=#4a2080;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="1050" y="420" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="ball-weights" value="IModelWeights" style="shape=ellipse;html=1;fillColor=#fff3e8;strokeColor=#f09040;strokeWidth=2;labelPosition=center;verticalLabelPosition=bottom;align=center;verticalAlign=top;spacingTop=5;fontStyle=1;fontSize=11;" vertex="1" parent="1">
          <mxGeometry x="1055" y="425" width="10" height="10" as="geometry" />
        </mxCell>
        <mxCell id="link-vit-weights" value="weights" style="endArrow=none;strokeColor=#4a2080;strokeWidth=1.5;dashed=1;fontSize=11;" edge="1" source="c-vit" target="sock-weights" parent="1">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="1000" y="480" />
              <mxPoint x="1000" y="430" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="link-siglip-weights" value="weights" style="endArrow=none;strokeColor=#9060c0;strokeWidth=1.5;dashed=1;fontSize=11;" edge="1" source="c-siglip" target="sock-weights" parent="1">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="1000" y="580" />
              <mxPoint x="1000" y="430" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="link-smogy-weights" value="weights" style="endArrow=none;strokeColor=#9060c0;strokeWidth=1.5;dashed=1;fontSize=11;" edge="1" source="c-smogy" target="sock-weights" parent="1">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="1000" y="680" />
              <mxPoint x="1000" y="430" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="link-hf-weights" style="endArrow=none;strokeColor=#f09040;strokeWidth=1.5;" edge="1" source="c-hf" target="ball-weights" parent="1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
\n"""
new_content.append(new_xml)
new_content.extend(lines[285:])

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_content)
