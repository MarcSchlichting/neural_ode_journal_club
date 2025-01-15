from manimlib import *

class Presentation(InteractiveScene):
    def construct(self):
        paper_title = TexText(r"Inferring Latent Dynamics Underlying Neural\\Population Activity via Neural Differential Equations").move_to(0.5 * UP)
        authors = TexText(r"Timothy D. Kim, Thomas Z. Luo, Jonathan W. Pillow, Carlos D. Brody",font_size=20).move_to(0.5 * DOWN)
        orchestra_img = ImageMobject("./figures/orchestra.jpg").move_to(UP)
        orchstra_cedit = TexText(r"Image Source: Berliner Philharmoniker", font_size=12).to_edge(DOWN+RIGHT, buff=0.2)
        orchestra_latent_text = TexText(r"Like musicians in an orchestra, neurons\\follow underlying \textbf{latent} dynamics.").move_to(2 * DOWN)
        latent_explanation = TexText(r"something that is hidden or not immediateley obvious", font_size=24).move_to(DOWN+LEFT)
        # explanation_path = Line([-1,-1,0],[1,1,0])
        # explanation_path = CubicBezier(get_smooth_cubic_bezier_handle_points([[0,0,0],[1,1,0],[3,5,0]]))
        arr = Arrow([-1,-0.5,0], [1,0.5,0])
        
        motivation_texts  = [TexText(r"This a mathematically dense paper."), TexText(r"The introduced models might be helpful\\for a lot of you in systems neuroscience."), TexText(r"We will build the modeling framework from the end.").move_to(2*DOWN)]
        
        hh_vm = np.loadtxt("./data/hh_simulation_vm.txt")[:,::8]
        hh_t = np.loadtxt("./data/hh_simulation_t.txt")[:,::8]
        
        
        # Title Slide
        self.wait()
        self.play(LaggedStart(Write(paper_title), Write(authors), lag_ratio=1))
        self.wait()
        self.play(LaggedStart(Uncreate(authors[::-1]), Uncreate(paper_title),lag_ratio=0.5))
        
        # Slide with Orchestra Picture
        self.play(LaggedStart(FadeIn(orchestra_img), FadeIn(orchstra_cedit)))
        self.play(Write(orchestra_latent_text))
        self.wait()
        self.play(LaggedStart(FadeOut(orchestra_img),FadeOut(orchstra_cedit), MoveAlongPath(orchestra_latent_text, Line(2 * DOWN, UP))))
        self.play(LaggedStart(orchestra_latent_text[50:56].animate.set_color(RED), ShowCreation(arr)))
        self.play(Write(latent_explanation))
        self.wait()
        self.play(LaggedStart(Uncreate(orchestra_latent_text), Uncreate(latent_explanation), FadeOut(arr)))
        
        # Motivation Slide
        self.play(Write(motivation_texts[0]))
        self.wait()
        self.play(LaggedStart(motivation_texts[0].animate.move_to(2*UP), Write(motivation_texts[1]),lag_ratio=0.5))
        self.wait()
        self.play(Write(motivation_texts[2]))
        self.wait()
        self.play(LaggedStart(*[Uncreate(mt) for mt in motivation_texts]))
        
        # # Spike Recordings
        # ax = [Axes([0,hh_t.max()],[hh_vm.min(), hh_vm.max()],height=0.4,width=6).to_edge(UP,buff=1).to_edge(LEFT,buff=1)]
        # # ax[0].get_y_axis_label("V_0",edge=[0,0,0])
        # xs = [np.stack([hh_t[0],hh_vm[0]],axis=1)]
        # for i in range(1,hh_vm.shape[0]):
        #     ax.append(Axes([0,hh_t.max()],[hh_vm.min(), hh_vm.max()],height=0.4,width=6).next_to(ax[-1],DOWN))
        #     xs.append(np.stack([hh_t[i],hh_vm[i]],axis=1))
        # self.play(LaggedStart([Write(TexText("Neuronal Voltage Recordings",font_size=20).next_to(ax[0],UP))]+[Write(Tex(fr"V_{{{i+1}}}",font_size=20).next_to(a,LEFT)) for i,a in enumerate(ax)]))
        # self.play(*[ShowCreation(VMobject().set_points(ax[i].c2p(*xs[i].T)).set_stroke(BLUE_B, 2),rate_func=linear,run_time=10) for i in range(len(xs))])

        
        
        # Poisson Process
        poisson_process_title = TexText("Nonhomogeneous Poisson Processes").to_edge(UP+LEFT)
        poisson_tex_one = Tex(R"""
                          t_{1}, t_{2}, t_{3},\ldots, t_{N} \sim \mathrm{PoissonProcess}(\lambda(t))
                          """)
        poisson_tex_full = Tex(R"""
                          \begin{matrix}
                          t_{1}^{(1)}, &  t_{2}^{(1)}, & t_{3}^{(1)}, & \ldots, &  t_{N_1}^{(1)} \\
                          t_{1}^{(2)}, & t_{2}^{(2)}, & t_{3}^{(2)}, & \ldots, & t_{N_2}^{(2)} \\
                          t_{1}^{(3)}, & t_{2}^{(3)}, & t_{3}^{(3)}, & \ldots, & t_{N_3}^{(3)} \\
                            & &   \vdots \\
                          t_{1}^{(M)}, & t_{2}^{(M)}, & t_{3}^{(M)}, & \ldots, & t_{N_M}^{(M)}
                          \end{matrix} \sim \begin{bmatrix} \mathrm{PoissonProcess}(\lambda_1(t))\vphantom{0^{(0)}_0} \\ 
                           \mathrm{PoissonProcess}(\lambda_2(t))\vphantom{0^{(0)}_0} \\
                           \mathrm{PoissonProcess}(\lambda_3(t))\vphantom{0^{(0)}_0} \\
                               \vdots  \\
                            \mathrm{PoissonProcess}(\lambda_M(t)) \vphantom{0^{(0)}_0}
                           \end{bmatrix}
                          """, font_size=32)
        spike_times = TexText("Spike Times",font_size=24).next_to(np.array([-2,-2.5,0]),LEFT)
        spike_time_curves = [CubicBezier(np.array([-2,-2.5,0]), np.array([2,-2.5,0]),poisson_tex_one[i].get_bottom() + DOWN, poisson_tex_one[i].get_bottom() +0.2 *DOWN).set_stroke(WHITE,2) for i in [0,3,6,13]]
        firing_rate = TexText("Firing Rate", font_size=24).next_to(np.array([3,2,0]),RIGHT)
        firing_rate_curve = CubicBezier(np.array([3,2,0]), np.array([1,2,0]), poisson_tex_one[31].get_top()+0.8*UP,poisson_tex_one[31].get_top()+0.2*UP).set_stroke(WHITE,2)
        num_neurons_group = Group()
        num_neurons_group.add(Arrow(poisson_tex_full.get_corner(UP+LEFT)+0.5*LEFT,poisson_tex_full.get_corner(DOWN+LEFT)+0.5*LEFT, buff=0).set_fill(GREY))
        num_neurons_group.add(TexText("Number of Neurons",font_size=24, t2c={"Number of Neurons":GREY}).rotate(np.pi/2).next_to(num_neurons_group[0],LEFT))
        
        
        self.play(Write(poisson_process_title))
        self.wait()
        self.play(Write(poisson_tex_one))
        self.wait()
        self.play(LaggedStart(Write(spike_times[::-1]),*[ShowCreation(stc) for stc in spike_time_curves]))
        self.wait()
        self.play(LaggedStart(Write(firing_rate),ShowCreation(firing_rate_curve)))
        self.wait()
        self.play(LaggedStart(Uncreate(spike_times),
                              *[Uncreate(stc) for stc in spike_time_curves],
                              Uncreate(firing_rate[::-1]),
                              Uncreate(firing_rate_curve),
                              Transform(poisson_tex_one, poisson_tex_full)))
        self.play(ShowCreation(num_neurons_group))
        self.wait()
        self.play(LaggedStart(Uncreate(num_neurons_group),
                  Uncreate(poisson_tex_one)))
        
        # From Firing Rate to Latent Variables
        fr_to_lv_title = TexText("From Firing Rates to Latent Variables").to_edge(UP+LEFT)
        exp_aff_map_short = Tex(R"""
                          \mathbf{\lambda} = \mathrm{exp}\left(\mathbf{A}\mathbf{z}(t) + \mathbf{b}\right)
                          """)
        exp_aff_map_long = Tex(R"""
                          \begin{bmatrix}
                          \lambda^{(1)}(t)\\
                          \lambda^{(2)}(t)\\
                          \lambda^{(3)}(t)\\
                          \vdots \\
                          \lambda^{(M)}(t)\\    
                          \end{bmatrix} = \mathrm{exp}\left(\begin{bmatrix}a_{11} & a_{12} & a_{13} & \ldots & a_{1K}\\
                              a_{21} & a_{22} & a_{23} &\ldots & a_{2K}\\
                                  a_{31} & a_{32} & a_{33} &\ldots & a_{3K}\\
                                 \vdots & \vdots & \vdots & \ddots & \vdots \\
                              a_{M1} & a_{M2} & a_{M3} & \ldots & a_{MK}
                              \end{bmatrix}\begin{bmatrix}z_1(t) \\ z_2(t) \\ z_3(t) \\ \vdots \\ z_K(t)\end{bmatrix} + 
                              \begin{bmatrix}b_1 \\ b_2 \\ b_3 \\ \vdots \\ b_K\end{bmatrix}\right)
                          """, font_size=32)
        dim_lambda = Tex(R"\mathrm{Firing~Rates}\in\mathbb{R}^{M}",font_size=24).next_to(np.array([-4.5,-2.5,0]),RIGHT)
        dim_lambda_curve = CubicBezier(np.array([-4.5,-2.5,0]),np.array([-4.5,-2.5,0])+LEFT, exp_aff_map_long[33].get_bottom()+0.8 * DOWN, exp_aff_map_long[33].get_bottom()+0.2 * DOWN).set_stroke(WHITE,2)
        dim_A = Tex(R"\mathrm{Loading~Matrix}\in\mathbb{R}^{M\times K}",font_size=24).next_to(np.array([-1,2.5,0]),RIGHT)
        dim_A_curve = CubicBezier(np.array([-1,2.5,0]),np.array([-1,2.5,0])+2 * LEFT,exp_aff_map_long.get_top()+0.8*UP,exp_aff_map_long.get_top()+0.2*UP).set_stroke(WHITE,2)
        dim_z = Tex(R"\mathrm{Latent~Variables}\in\mathbb{R}^{K}",font_size=24).next_to(np.array([3.2,-2.75,0]),LEFT)
        dim_z_curve = CubicBezier(np.array([3.2,-2.75,0]),np.array([3.2,-2.75,0])+RIGHT,np.array([2.8,-1.2,0])+DOWN,np.array([2.8,-1.2,0])).set_stroke(WHITE,2)
        dim_b = Tex(R"\mathrm{Bias~Vector}\in\mathbb{R}^{K}",font_size=24).next_to(np.array([4,2,0]),RIGHT)
        dim_b_curve = CubicBezier(np.array([4,2,0]),np.array([4,2,0])+LEFT,np.array([4.2,1.2,0])+0.5 * UP, np.array([4.2,1.2,0])).set_stroke(WHITE,2)
        exp_expl = Tex(R"\mathrm{Firing~Rates}\in\mathbb{R}^+",font_size=24).next_to(np.array([-3,2,0]),LEFT)
        exp_expl_curve = CubicBezier(np.array([-3,2,0]), np.array([-3,2,0])+RIGHT, np.array([-2.9,0.2,0])+UP, np.array([-2.9,0.2,0])).set_stroke(WHITE,2)
        
        self.play(Transform(poisson_process_title,fr_to_lv_title))
        self.wait()
        self.play(Write(exp_aff_map_short))
        self.wait()
        self.play(Transform(exp_aff_map_short,exp_aff_map_long),run_time=2)
        self.wait()
        self.play(LaggedStart(Write(dim_lambda),ShowCreation(dim_lambda_curve)))
        self.wait()
        self.play(LaggedStart(Write(dim_z[::-1]), ShowCreation(dim_z_curve)))
        self.wait()
        self.play(LaggedStart(Write(dim_A),ShowCreation(dim_A_curve)))
        self.wait()
        self.play(LaggedStart(Write(dim_b),ShowCreation(dim_b_curve)))
        self.wait()
        self.play(LaggedStart(Write(exp_expl[::-1]), ShowCreation(exp_expl_curve)))
        self.wait()        
        self.play(LaggedStart(Uncreate(dim_lambda),
                              Uncreate(dim_lambda_curve),
                              Uncreate(dim_z[::-1]),
                              Uncreate(dim_z_curve),
                              Uncreate(dim_A),
                              Uncreate(dim_A_curve),
                              Uncreate(dim_z),
                              Uncreate(dim_z_curve),
                              Uncreate(dim_b),
                              Uncreate(dim_b_curve),
                              Uncreate(exp_expl[::-1]),
                              Uncreate(exp_expl_curve),
                              Uncreate(exp_aff_map_short)))
        
        
        # Latent Dynamics
        latent_dynamics_title = TexText("Latent Dynamics").to_edge(UP+LEFT)
        latent_dynamic_intro_text = TexText(r"Given a latent state $\mathbf{z}$, what\\is the rate of change of the latent variables?")
        latent_ode = Tex(R"\dfrac{\mathrm{d}\mathbf{z}}{\mathrm{d}t} = f(\mathbf{z}(t),\mathbf{u}(t),t)")
        latent_change = TexText("Latent Variables Rate of Change",font_size=24).next_to(np.array([-2,2,0]),LEFT)
        latent_change_curve = CubicBezier(np.array([-2,2,0]),np.array([-2,2,0])+2 *RIGHT,np.array([-1.7,0.6,0])+UP,np.array([-1.7,0.6,0])).set_stroke(WHITE,2)
        f_exp = TexText("An Arbitrary Non-Linear Function",font_size=24).next_to(np.array([-2,-1.5,0]),LEFT)
        f_exp_curve = CubicBezier(np.array([-2,-1.5,0]),np.array([-2,-1.5,0])+RIGHT,np.array([-0.75,-0.4,0])+DOWN,np.array([-0.75,-0.4,0])).set_stroke(WHITE,2)
        z_exp = TexText(r"\begin{flushleft}Current Values\\of Latent Variables\end{flushleft}",font_size=24).next_to(np.array([1,2.5,0]),RIGHT)
        z_exp_curve = CubicBezier(np.array([1,2.5,0]),np.array([1,2.5,0])+2 * LEFT,np.array([-0.25,0.2,0])+UP, np.array([-0.25,0.2,0])).set_stroke(WHITE,2)
        t_exp = TexText("Time",font_size=24).next_to(np.array([1.5,-2,0]),RIGHT)
        t_exp_curve = CubicBezier(np.array([1.5,-2,0]),np.array([1.5,-2,0])+LEFT, np.array([1.8,-0.25,0])+DOWN, np.array([1.8,-0.25,0])).set_stroke(WHITE,2)
        u_exp = TexText("Input",font_size=24).next_to(np.array([3.5,1.5,0]),RIGHT)
        u_exp_curve = CubicBezier(np.array([3.5,1.5,0]),np.array([3.5,1.5,0])+3.5 * LEFT,np.array([0.8,0.2,0])+UP,np.array([0.8,0.2,0])).set_stroke(WHITE,2)
        ode_text = TexText(r"\textbf{O}rdinary \textbf{D}ifferential \textbf{E}quation",t2c={r"\textbf{O}rdinary \textbf{D}ifferential \textbf{E}quation":RED}).move_to(np.array([0,2,0])).to_edge(LEFT)
        example_ode = TexText(r"Example: $\frac{\mathrm{d}z}{\mathrm{d}t}=\mathrm{cos}(t)\quad z(0)=0$").move_to(np.array([0,1,0])).to_edge(LEFT)
        ax_ode_example = Axes([0,3],[0,1.5],width=8,height=3).move_to([0,-1.5,0])
        ax_ode_example_x_label = ax_ode_example.get_x_axis_label(r"t")
        ax_ode_example_y_label = ax_ode_example.get_y_axis_label(r"z")
        t_example = ValueTracker(0)
        solution_example = VMobject().set_points(ax_ode_example.c2p(*np.array([0,0,0])))
        values = Group()
        values.add(Tex(r"t=").move_to([0,0,0]))
        values.add(DecimalNumber(text_config={"font":"Times New Roman"}).next_to(values[0],RIGHT))
        values.add(Tex(r"\dfrac{\mathrm{d}z}{\mathrm{d}t}=").move_to([2.5,0,0]))
        values.add(DecimalNumber(text_config={"font":"Times New Roman"}).next_to(values[2],RIGHT))
        
        def update_glow_dot(gd):
            t = t_example.get_value()
            t_span = np.linspace(0,t,201)
            z_span = np.sin(t_span)
            sol = np.stack([t_span,z_span, np.zeros_like(t_span)],axis=0)
            sol_p = ax_ode_example.c2p(*sol)
            # dt = 0.2
            curr_z = np.array([t,np.sin(t),0])
            start = np.array([curr_z[0]-0.2,curr_z[1]-0.2*np.cos(curr_z[0]),0])
            end = np.array([curr_z[0]+0.2,curr_z[1]+0.2*np.cos(curr_z[0]),0])
            start_p =ax_ode_example.c2p(*start)
            end_p = ax_ode_example.c2p(*end)
            euler_line.set_points_by_ends(start=start_p,end=end_p)
            gd.set_points(ax_ode_example.c2p(*curr_z))
            solution_example.set_points(sol_p)
            values[1].set_value(curr_z[0])
            values[3].set_value(np.cos(curr_z[0]))
        

            
        euler_line_start = ax_ode_example.c2p(*np.array([-0.2,-0.2,0]))
        euler_line_end = ax_ode_example.c2p(*np.array([0.2,0.2,0]))
        euler_line = Line(euler_line_start,euler_line_end).set_stroke(BLUE)
        gd_ode_example = GlowDot(ax_ode_example.c2p(*[0,0,0]),color=RED)
        # gd_ode_example = Dot(radius=0.2, color=RED).move_to(ax_ode_example.c2p(*[0,0,0]))
        gd_ode_example.add_updater(update_glow_dot)
        
        self.play(Transform(poisson_process_title,latent_dynamics_title))
        # self.add(latent_dynamics_title)
        self.wait()
        self.play(Write(latent_dynamic_intro_text))
        self.wait()
        self.play(Transform(latent_dynamic_intro_text,latent_ode))
        self.wait()
        self.play(LaggedStart(Write(latent_change[::-1]),ShowCreation(latent_change_curve)))
        self.wait()
        self.play(LaggedStart(Write(f_exp[::-1]),ShowCreation(f_exp_curve)))
        self.wait()
        self.play(LaggedStart(Write(z_exp),ShowCreation(z_exp_curve)))
        self.wait()
        self.play(LaggedStart(Write(t_exp),ShowCreation(t_exp_curve)))
        self.wait()
        self.play(LaggedStart(Write(u_exp),ShowCreation(u_exp_curve)))
        # self.add(NumberPlane())
        self.wait()
        self.play(LaggedStart(Uncreate(latent_change[::-1]),
                              Uncreate(latent_change_curve),
                              Uncreate(f_exp[::-1]),
                              Uncreate(f_exp_curve),
                              Uncreate(z_exp),
                              Uncreate(z_exp_curve),
                              Uncreate(t_exp),
                              Uncreate(t_exp_curve),
                              Uncreate(u_exp),
                              Uncreate(u_exp_curve),
                              latent_dynamic_intro_text.animate.move_to(np.array([3,2,0])),
                              Write(ode_text)))
        self.wait()
        self.play(LaggedStart(Write(example_ode),
                              ShowCreation(ax_ode_example),
                              Write(ax_ode_example_x_label),
                              Write(ax_ode_example_y_label),
                              ShowCreation(values),
                              ShowCreation(solution_example),
                              ShowCreation(euler_line)))
        self.add(gd_ode_example)
        
        for v in np.linspace(0.25,3,9):
            self.wait(1)
            self.play(t_example.animate.set_value(v),run_time=1)
        self.wait(1)
        
        self.play(LaggedStart(FadeOut(example_ode),
                              FadeOut(ax_ode_example),
                              FadeOut(ax_ode_example_x_label),
                              FadeOut(ax_ode_example_y_label),
                              FadeOut(values),
                              FadeOut(solution_example),
                              FadeOut(euler_line),
                              FadeOut(gd_ode_example)))
        
        # self.add(ode_text)
        main_contribution = Group()
        main_contribution.add(TexText(r"Main Contribution of the Paper").move_to([0,0,0]))
        main_contribution.add(Tex(r"\underbrace{f(\mathbf{z},t,u)=\mathrm{Artificial~Neural~Network}}_{\mathrm{Neural~ODE}}").move_to([0,-2,0]))
        main_contributions_box = RoundedRectangle(
            width=main_contribution.get_width() + 0.5,  # Add padding
            height=main_contribution.get_height() + 0.5,  # Add padding
            corner_radius=0.2,
            color=RED,
            fill_color=RED_E,  # Fill color for the box
            fill_opacity=0.2  # Set opacity for shading (0 is fully transparent,
        ).move_to(main_contribution.get_center())
        self.wait()
        self.play(LaggedStart(ShowCreation(main_contributions_box),
                              Write(main_contribution[0]),
                              Write(main_contribution[1]),
                              ))
        self.wait()
        self.play(LaggedStart(Uncreate(main_contributions_box),
                              Uncreate(main_contribution[0]),
                            #   Uncreate(main_contribution[1]),
                              Uncreate(latent_dynamic_intro_text[::-1]),
                              Uncreate(ode_text),
                              Uncreate(poisson_process_title[::-1])))
        
        # Phase Diagram
        phase_dig_title = TexText("Latent Dynamics Interpretability")
        def func(Z,z2=None):
            # Example parameters
            a = 0
            b = 0
            rho = 2
            tau = 15
            if z2 is None:
                z1 = Z[0]
                z2 = Z[1]
            else:
                z1 = Z
                z2 = z2
            
            # Compute derivatives
            z1_dot = rho * tau * (z1 - 1/3 * z1**3 - z2)
            z2_dot = tau / rho * (z1 + a - b * z2)
            
            return 0.01 * np.array([z1_dot,z2_dot])

        # Create the number plane
        ax_pd_pt = NumberPlane((-4, 4), (-2, 2), faded_line_ratio=1, height=5,width=10).move_to(0.5 * DOWN)
        ax_pd = NumberPlane((-4.5, 4.5), (-2.5, 2.5), faded_line_ratio=1, height=5*5/4,width=10*9/8).move_to(0.5 * DOWN)
        ax_pd.background_lines.set_stroke(BLUE, 0)
        ax_pd.faded_lines.set_stroke(GREY, 0.5, 0.5)
        ax_pd_x_label = Tex("z_1",font_size=32).next_to(ax_pd.x_axis.get_end(), 0.7 * RIGHT)
        ax_pd_y_label = Tex("z_2",font_size=32).next_to(ax_pd.y_axis.get_end(), 0.1 * TOP)

        # Generate streamlines
        stream_lines = StreamLines(func, 
                                   ax_pd_pt,
                                   color_by_magnitude=True,
                                   density=5,
                                   n_samples_per_line=1,
                                   solution_time=2,
                                   magnitude_range=[0,6],
                                   stroke_color=WHITE,
                                    stroke_width=2,
                                    stroke_opacity=1,
                                    color_map="BuPu"
                                   )
        animated_lines = AnimatedStreamLines(stream_lines,rate_multiple=1,line_anim_config=dict(time_width=3))
        
        self.play(Transform(main_contribution[1], phase_dig_title))
        self.wait()
        self.play(main_contribution[1].animate.to_edge(LEFT+UP))
        # self.add(phase_dig_title.to_edge(LEFT+UP))
        self.add(stream_lines)
        self.add(animated_lines)
        self.wait()
        self.play(LaggedStart(ShowCreation(ax_pd),
                              Write(ax_pd_x_label),
                              Write(ax_pd_y_label)))
        self.wait()
        self.play(LaggedStart(Uncreate(ax_pd),
                              Uncreate(ax_pd_x_label),
                              Uncreate(ax_pd_y_label)))
        self.play(*[Uncreate(vm) for vm in animated_lines])
        self.play(Uncreate(main_contribution[1][::-1]))
        # self.wait(5)
        
        
        # Overview Poisson Latent Neural Differential Equations for Spiking Neural Data
        plnde_title = TexText("Method Overview").to_edge(UP+LEFT)
        
        plnde_neural_ode_eq = Tex(R"z(t)=\int\mathrm{ANN}(z(t),u(t),t)~\mathrm{d}t",font_size=24)
        plnde_neural_ode_cap = TexText(r"\textbf{Latent Dynamics}",font_size=32).next_to(plnde_neural_ode_eq,DOWN)
        plnde_neural_ode_group = Group()
        plnde_neural_ode_group.add(plnde_neural_ode_eq)
        plnde_neural_ode_group.add(plnde_neural_ode_cap)
        box_neural_ode = RoundedRectangle(plnde_neural_ode_group.get_width()+0.5,plnde_neural_ode_group.get_height()+0.5,corner_radius=0.25).move_to(plnde_neural_ode_group.center())
        plnde_neural_ode_group.add(box_neural_ode)
        plnde_neural_ode_group.to_edge(LEFT)
        
        plnde_mapping_eq = Tex(R"\lambda(t)=\mathrm{exp}(Az(t)+b)",font_size=24)
        plnde_mapping_cap = TexText(r"\textbf{Mapping}",font_size=32).next_to(plnde_mapping_eq,DOWN)
        plnde_mapping_group = Group()
        plnde_mapping_group.add(plnde_mapping_eq)
        plnde_mapping_group.add(plnde_mapping_cap)
        box_mapping = RoundedRectangle(plnde_mapping_group.get_width()+0.5,plnde_neural_ode_group.get_height(),corner_radius=0.25).move_to(plnde_mapping_group.center())
        plnde_mapping_group.add(box_mapping)
        
        plnde_spike_eq = Tex(R"t_1,\ldots,t_N\sim\mathrm{PoissonProcess(\lambda(t))}",font_size=24)
        plnde_spike_cap = TexText(r"\textbf{Spike Train}",font_size=32).next_to(plnde_spike_eq,DOWN)
        plnde_spike_group = Group()
        plnde_spike_group.add(plnde_spike_eq)
        plnde_spike_group.add(plnde_spike_cap)
        box_spike = RoundedRectangle(plnde_spike_group.get_width()+0.5,plnde_neural_ode_group.get_height(),corner_radius=0.25).move_to(plnde_spike_group.center())
        plnde_spike_group.add(box_spike)
        plnde_spike_group.to_edge(RIGHT)
        
        arr_neural_ode_map = Arrow(box_neural_ode.get_edge_center(RIGHT)+np.array([0,0.25,0]), box_mapping.get_edge_center(LEFT)+np.array([0,0.25,0]),buff=0,path_arc=-0.8)
        arr_map_spike = Arrow(box_mapping.get_edge_center(RIGHT)+np.array([0,0.25,0]), box_spike.get_edge_center(LEFT)+np.array([0,0.25,0]),buff=0,path_arc=-0.8)
        arr_map_neural_ode = Arrow( box_mapping.get_edge_center(LEFT)+np.array([0,-0.25,0]),box_neural_ode.get_edge_center(RIGHT)+np.array([0,-0.25,0]),buff=0,path_arc=-0.8)
        arr_spike_map = Arrow(box_spike.get_edge_center(LEFT)+np.array([0,-0.25,0]),box_mapping.get_edge_center(RIGHT)+np.array([0,-0.25,0]),buff=0,path_arc=-0.8)
        
        arr_generation = Arrow(box_neural_ode.get_edge_center(LEFT)+1.5*UP, box_spike.get_edge_center(RIGHT)+1.5*UP,buff=0).set_fill(GREY).set_stroke(GREY,0)
        arr_generation_label = TexText(R"Generative Process", font_size=24,t2c={"Generative Process":GREY}).move_to(arr_generation.get_center()+0.25*UP)
        
        arr_inference = Arrow(box_spike.get_edge_center(RIGHT)+1.5*DOWN,box_neural_ode.get_edge_center(LEFT)+1.5*DOWN,buff=0).set_fill(GREY).set_stroke(GREY,0)
        arr_inference_label = TexText(R"Inference", font_size=24,t2c={"Inference":GREY}).move_to(arr_inference.get_center()+0.25*DOWN)
        
        plnde_brace = Brace(Group([box_mapping,box_neural_ode,box_spike]),DOWN).move_to(2.5 * DOWN)
        plnde_brace_cap = TexText('Poisson Latent Neural Differential Equations',font_size=32).next_to(plnde_brace,DOWN)
        
        self.play(Write(plnde_title))
        # self.add(plnde_title)
        self.wait()
        self.play(LaggedStart(*[ShowCreation(gc) for gc in plnde_neural_ode_group],
                              ))
        self.wait()
        self.play(LaggedStart(GrowArrow(arr_neural_ode_map),
                              *[ShowCreation(gc) for gc in plnde_mapping_group]))
        self.wait()
        self.play(LaggedStart(GrowArrow(arr_map_spike),
                              *[ShowCreation(gc) for gc in plnde_spike_group]))
        self.wait()
        self.play(LaggedStart(Write(arr_generation_label),
                              GrowArrow(arr_generation)))
        self.wait()
        self.play(LaggedStart(Write(arr_inference_label),
                              GrowArrow(arr_inference)))
        self.wait()
        self.play(LaggedStart(GrowArrow(arr_spike_map),
                              GrowArrow(arr_map_neural_ode),lag_ratio=0.8))
        self.wait()
        self.play(LaggedStart(ShowCreation(plnde_brace),
                              Write(plnde_brace_cap)))
        self.wait()
        
        self.play(LaggedStart(*[Uncreate(gc) for gc in plnde_neural_ode_group],
                              FadeOut(arr_neural_ode_map),
                              *[Uncreate(gc) for gc in plnde_mapping_group],
                              FadeOut(arr_map_spike),
                              *[Uncreate(gc) for gc in plnde_spike_group],
                              FadeOut(arr_generation_label),
                              FadeOut(arr_generation),
                              FadeOut(arr_inference_label),
                              FadeOut(arr_inference),
                              FadeOut(arr_spike_map),
                              FadeOut(arr_map_neural_ode),
                              Uncreate(plnde_brace),
                              Uncreate(plnde_brace_cap),
                              Uncreate(plnde_title)))
        
        
        
        
        # Experiments
        experiments_title = TexText("Experiments")
        experiments_synth = TexText("Synthetic Experiments").to_edge(UP+LEFT)
        experiments_animal = TexText("Animal Experiments").to_edge(UP+LEFT)
        fig2 = ImageMobject("./figures/fig2_cropped_dark.png",height=6).move_to([0,-0.5,5])
        fig4 = ImageMobject("./figures/fig4_cropped_dark.png",height=4).move_to([-4,-5,5])
        fig1 = ImageMobject("./figures/fig1_cropped_dark.png",height=2).move_to([-10,5,5])
        fig4_annot_no_neurons = TexText(R"101 M2 neurons\\99 Cg1 neurons\\249 PrL neurons\\47 MO neurons",font_size=24).move_to([-0.5,2.5,0])
        fig4_annot_box = RoundedRectangle(fig4_annot_no_neurons.get_width()+0.4,fig4_annot_no_neurons.get_height()+0.4,corner_radius=0.22).set_stroke(WHITE,2).move_to(fig4_annot_no_neurons.get_center())
        fig4_annot_path = CubicBezier(fig4_annot_box.get_edge_center(DOWN),fig4_annot_box.get_edge_center(DOWN)+DOWN,np.array([-0.7,0.4,0])+UP, np.array([-0.7,0.4,0])).set_stroke(WHITE,2)
        
        self.play(Write(experiments_title))
        self.wait()
        self.play(Transform(experiments_title,experiments_synth))
        self.play(LaggedStart(FadeIn(fig2),
                              fig2.animate.move_to([0,-0.5,0])))
        self.wait()
        self.play(LaggedStart(Transform(experiments_title,experiments_animal),
                              Rotate(fig2, angle=PI/2, axis=UP, about_point=RIGHT * FRAME_WIDTH/2, run_time=2),
                              fig2.animate.move_to([0,-0.5,5]),
                              FadeOut(fig2),
                              FadeIn(fig4),
                              fig4.animate.move_to([2,-0.5,0]),
                              FadeIn(fig1),
                              fig1.animate.move_to([-4.5,-0.5,0])))
        self.wait()
        self.play(LaggedStart(Write(fig4_annot_no_neurons),
                              ShowCreation(fig4_annot_box), 
                              ShowCreation(fig4_annot_path)))
        self.wait()
        self.play(LaggedStart(Uncreate(fig4_annot_path),
                              Uncreate(fig4_annot_box),
                              Uncreate(fig4_annot_no_neurons),
                              Rotate(fig4, angle=PI/2, axis=UP, about_point=RIGHT * FRAME_WIDTH/2, run_time=2),
                              fig4.animate.move_to([0,3,5]),
                              FadeOut(fig4),
                              Rotate(fig1, angle=PI/2, axis=UP, about_point=RIGHT * FRAME_WIDTH/2, run_time=2),
                              fig1.animate.move_to([0,-5,5]),
                              FadeOut(fig1)))
        
        # Pros and Cons of the Paper
        LIST_VERT_SEP = 0.3
        conclusions_title = TexText("Conclusions").to_edge(UP+LEFT)
        pros_title = TexText("Pros",font_size=32).move_to(LEFT_SIDE/2 + 2 * UP)
        pros_line = Line(LEFT_SIDE+np.array([0.5,1.7,0]), ORIGIN + np.array([-0.5,1.7,0])).set_stroke(WHITE,2)
        cons_title = TexText("Cons",font_size=32).move_to(RIGHT_SIDE/2 + 2 * UP)
        cons_line = Line(ORIGIN + np.array([0.5,1.7,0]),RIGHT_SIDE+np.array([-0.5,1.7,0])).set_stroke(WHITE,2)
        
        pros_list = Group(TexText(R"Phase portraits are very interpretable.", font_size=24).align_to(pros_line.get_start()+LIST_VERT_SEP*DOWN,UL))
        pros_list.add(TexText(R"Mathematically rirgorous paper with detailed derivations.", font_size=24).align_to(pros_list[-1].get_corner(LEFT+DOWN)+LIST_VERT_SEP*DOWN,UL))
        pros_list.add(TexText(R"\flushleft{Paper is well written and color scheme in figures\\is chosen consistently.}", font_size=24).align_to(pros_list[-1].get_corner(LEFT+DOWN)+LIST_VERT_SEP*DOWN,UL))
        pros_list.add(TexText(R"\flushleft{The discussion section of the paper discusses some\\limitations of their approach.}", font_size=24).align_to(pros_list[-1].get_corner(LEFT+DOWN)+LIST_VERT_SEP*DOWN,UL))
        
        cons_list = Group(TexText(R"\flushleft{Validation of their model is only possible for synthetic\\data as now ground truth data exists real data.}", font_size=24).align_to(cons_line.get_start()+LIST_VERT_SEP*DOWN,UL))
        cons_list.add(TexText(R"\flushleft{The model has no uncertainty measure (i.e., is\\a point estimate).}", font_size=24).align_to(cons_list[-1].get_corner(LEFT+DOWN)+LIST_VERT_SEP*DOWN,UL))
        cons_list.add(TexText(R"\flushleft{All 3 synthetic datasets have rather simple low-\\dimensional latent dynamics.}", font_size=24).align_to(cons_list[-1].get_corner(LEFT+DOWN)+LIST_VERT_SEP*DOWN,UL))
        cons_list.add(TexText(R"\flushleft{The number of neurons for the synthetic dataset\\appears arbitrary.}", font_size=24).align_to(cons_list[-1].get_corner(LEFT+DOWN)+LIST_VERT_SEP*DOWN,UL))
        cons_list.add(TexText(R"\flushleft{The published code only contains the code to run\\the simplest of the synthetic datasets.}", font_size=24).align_to(cons_list[-1].get_corner(LEFT+DOWN)+LIST_VERT_SEP*DOWN,UL))
        
        self.play(Transform(experiments_title, conclusions_title))
        # self.add(conclusions_title)
        self.wait()
        self.play(LaggedStart(ShowCreation(pros_line),
                              Write(pros_title),
                              ShowCreation(cons_line),
                              Write(cons_title)))
        
        for pl in pros_list:
            self.wait()
            self.play(Write(pl))
        
        for cl in cons_list:
            self.wait()
            self.play(Write(cl))
            
        
        
        
        
 
        
        
        